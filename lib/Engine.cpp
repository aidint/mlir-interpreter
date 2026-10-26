#include "interpreter/Engine.h"
#include "interpreter/Interfaces/EvaluableAttrInterface.h"

#include "mlir/IR/Matchers.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "interpreter"

namespace mlir::interpreter {

Answer EvalScope::query(Value value) { return engine.queryNested(value); }

SmallVector<Answer> EvalScope::walk(Region &region, ArrayRef<Answer> args) {
  return {};
}

bool EvalScope::isOrdered() const { return false; }

EvalValueStorageAllocator &EvalScope::getAllocator() {
  return engine.getQueryAllocator();
}

Answer Engine::query(Value value) {
  CacheOutcome outcome;
  return query(value, outcome);
}

Answer Engine::query(Value value, CacheOutcome &outcome) {
  assert(!queryAllocator &&
         "cannot start a query while another query is active");
  remaining = budget;
  ctx.getCache().beginQuery();
  answerAllocator.beginRetaining();
  queryAllocator.emplace(ctx, false);
  // Nothing is memoized yet, so `value` goes straight to `compute`, which
  // reports the lookup of its operation.
  outcome = CacheOutcome::None;
  Answer answer = compute(value, &outcome);
  if (answer.completed())
    if (const auto &value = answer.getValue())
      answer.value = answerAllocator.retain(*value);
  queryAllocator.reset();
  queryAnswers.clear();
  return answer;
}

Answer Engine::queryNested(Value value) {
  assert(queryAllocator && "nested query requires an active query");
  return evaluate(value);
}

Answer Engine::evaluate(Value value) {
  if (auto it = queryAnswers.find(value); it != queryAnswers.end())
    return it->second;
  Answer answer = compute(value);
  if (answer.completed())
    queryAnswers.try_emplace(value, answer);
  return answer;
}

Answer Engine::compute(Value value, CacheOutcome *outcome) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return Answer::unknown();

  Operation *op = result.getOwner();
  auto evaluable = dyn_cast<EvaluableOpInterface>(op);
  if (!evaluable) {
    if (!op->hasTrait<OpTrait::ConstantLike>()) {
      LDBG() << "no evaluation function for '" << op->getName() << "'";
      return Answer::unknown();
    }
    // A null value must not reach `Answer`, which would hold it as known.
    if (EvalValue value = evaluateConstant(op, outcome))
      return value;
    return Answer::unknown();
  }

  // Cache-key methods and graph-region operands can re-enter an operation
  // whose evaluation is in progress.
  if (!active.insert(op).second)
    return EvalStatus::Cycle;
  llvm::scope_exit leave([&] { active.erase(op); });

  EvalScope scope(*this);
  // Eligibility may query operands, so it runs inside the cycle guard and
  // shares the query's budget and memo with hashing and evaluation.
  auto eligible = evaluable.isCacheable(scope);
  if (!eligible.completed())
    return eligible.status;
  bool cacheable = eligible.getValue();

  std::optional<detail::CacheLookup> cached;
  if (cacheable) {
    auto lookup = ctx.getCache().lookup(op, scope);
    if (!lookup.completed())
      return lookup.status;
    cached = lookup.getValue();
    // An entry can hold an answer for a sibling result only, which is a miss.
    std::optional<Answer> hit;
    if (cached->entry)
      hit = cached->entry->results[result.getResultNumber()];
    if (outcome)
      *outcome = hit ? CacheOutcome::Hit : CacheOutcome::Miss;
    if (hit)
      return *hit;
  }

  // The step is charged before `evaluate` queries the operands it needs.
  if (remaining == 0)
    return EvalStatus::Exhausted;
  --remaining;

  SmallVector<Answer> results = evaluable.evaluate(scope);

  assert(results.size() == op->getNumResults() &&
         "evaluation must return one answer per operation result");
  detail::CacheEntry *entry = nullptr;
  if (cached && llvm::any_of(results, [](const Answer &result) {
        return result.completed();
      })) {
    entry = cached->entry;
    if (entry)
      ctx.getCache().fill(entry, results);
    else
      entry = ctx.getCache().insert(op, *cached, results, scope);
  }

  // Siblings were computed too; record them so querying one later in this
  // query doesn't evaluate `op` again.
  for (auto [i, computed] : llvm::enumerate(results)) {
    Answer sibling = computed;
    if (entry)
      if (const auto &slot = entry->results[i])
        sibling = *slot;
    if (sibling.completed())
      queryAnswers.try_emplace(op->getResult(i), sibling);
  }

  if (entry)
    if (const auto &slot = entry->results[result.getResultNumber()])
      return *slot;
  return results[result.getResultNumber()];
}

EvalValue Engine::evaluateConstant(Operation *op, CacheOutcome *outcome) {
  Attribute attr;
  if (!matchPattern(op, m_Constant(&attr)))
    return {};
  auto evaluable = dyn_cast<EvaluableAttrInterface>(attr);
  if (!evaluable) {
    LDBG() << "no interpreter value for attribute " << attr;
    return {};
  }
  if (!evaluable.isCacheable())
    return evaluable.toEvalValue(*queryAllocator);
  detail::EvalCache &cache = ctx.getCache();
  Type type = op->getResult(0).getType();
  bool hit;
  EvalValue value = cache.lookupConstant(attr, type, hit);
  if (outcome)
    *outcome = hit ? CacheOutcome::Hit : CacheOutcome::Miss;
  if (hit)
    return value;
  // Converting straight into the cache needs no copy. A null value is stored
  // too, so a constant without one isn't converted again.
  value = evaluable.toEvalValue(cache.allocator);
  cache.insertConstant(attr, type, value);
  return value;
}

FailureOr<func::FuncOp>
Engine::specialize(func::CallOp call, ArrayRef<std::optional<EvalValue>> args) {
  return failure();
}

StringRef stringifyEvalStatus(EvalStatus status) {
  switch (status) {
  case EvalStatus::Completed:
    return "completed";
  case EvalStatus::Exhausted:
    return "exhausted";
  case EvalStatus::NeedsOrder:
    return "needs order";
  case EvalStatus::Cycle:
    return "cycle";
  }
  llvm_unreachable("unhandled EvalStatus");
}

} // namespace mlir::interpreter
