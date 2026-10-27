#include "interpreter/Engine.h"
#include "interpreter/Interfaces/EvaluableAttrInterface.h"

#include "mlir/IR/Matchers.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "interpreter"

namespace mlir::interpreter {

BlockBinding::~BlockBinding() {
  if (block)
    engine.unbind(block);
}

Answer EvalScope::query(Value value) { return engine.queryNested(value); }

BlockBinding EvalScope::bind(Block *block, ArrayRef<Answer> args) {
  return BlockBinding(engine, engine.bind(block, args) ? block : nullptr);
}

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
      answer.value = answerAllocator.retain(value);
  assert(frames.empty() && bindings.empty() &&
         "every binding must end with the query");
  queryAllocator.reset();
  queryAnswers.clear();
  return answer;
}

Answer Engine::queryNested(Value value) {
  assert(queryAllocator && "nested query requires an active query");
  return evaluate(value);
}

bool Engine::bind(Block *block, ArrayRef<Answer> args) {
  assert(args.size() == block->getNumArguments() &&
         "binding requires one answer per block argument");
  assert(llvm::none_of(
             frames,
             [&](const BindingFrame &frame) { return frame.block == block; }) &&
         "a block cannot be bound twice at once");
  if (remaining == 0)
    return false;
  --remaining;
  frames.push_back({block, {}});
  for (auto [arg, answer] : llvm::zip(block->getArguments(), args))
    bindings.try_emplace(arg, answer);
  return true;
}

void Engine::unbind(Block *block) {
  assert(!frames.empty() && frames.back().block == block &&
         "bindings must end in reverse order");
  for (Value value : frames.back().memoized)
    queryAnswers.erase(value);
  for (BlockArgument arg : block->getArguments())
    bindings.erase(arg);
  frames.pop_back();
}

bool Engine::isInBoundRegion(Operation *op) const {
  return llvm::any_of(frames, [&](const BindingFrame &frame) {
    return frame.block->getParent()->isAncestor(op->getParentRegion());
  });
}

Answer Engine::evaluate(Value value) {
  if (auto it = queryAnswers.find(value); it != queryAnswers.end())
    return it->second;
  Answer answer = compute(value);
  if (answer.completed())
    memoize(value, answer);
  return answer;
}

void Engine::memoize(Value value, const Answer &answer) {
  if (!queryAnswers.try_emplace(value, answer).second)
    return;
  // A loop's body can't use its results, so the frames that contain the
  // definition were bound from outermost to innermost.
  Region *region = value.getParentRegion();
  auto frame = llvm::find_if(llvm::reverse(frames), [&](BindingFrame &frame) {
    return frame.block->getParent()->isAncestor(region);
  });
  if (frame != frames.rend())
    frame->memoized.push_back(value);
}

Answer Engine::compute(Value value, CacheOutcome *outcome) {
  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    // A bound argument holds the current iteration's value, and costs no step.
    auto it = bindings.find(cast<BlockArgument>(value));
    return it != bindings.end() ? it->second : Answer::unknown();
  }

  Operation *op = result.getOwner();
  auto evaluable = dyn_cast<EvaluableOpInterface>(op);
  if (!evaluable) {
    if (!op->hasTrait<OpTrait::ConstantLike>()) {
      LDBG() << "no evaluation function for '" << op->getName() << "'";
      return Answer::unknown();
    }
    return evaluateConstant(op, outcome);
  }

  // Cache-key methods and graph-region operands can re-enter an operation
  // whose evaluation is in progress.
  if (!active.insert(op).second)
    return EvalStatus::Cycle;
  llvm::scope_exit leave([&] { active.erase(op); });

  EvalScope scope(*this);
  // An op in a bound region may depend on the current bindings, which cache
  // keys can't tell apart across iterations, so it skips the cache.
  bool cacheable = false;
  if (!isInBoundRegion(op)) {
    // Eligibility may query operands, so it runs inside the cycle guard and
    // shares the query's budget and memo with hashing and evaluation.
    auto eligible = evaluable.isCacheable(scope);
    if (!eligible.completed())
      return eligible.status;
    cacheable = eligible.getValue();
  }

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
      memoize(op->getResult(i), sibling);
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
