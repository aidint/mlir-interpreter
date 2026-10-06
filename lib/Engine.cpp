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

namespace {

Answer evaluateConstant(Operation *op, EvalValueStorageAllocator &allocator) {
  Attribute attr;
  if (!matchPattern(op, m_Constant(&attr)))
    return {EvalStatus::Completed, std::nullopt};
  auto evaluable = dyn_cast<EvaluableAttrInterface>(attr);
  if (!evaluable) {
    LDBG() << "no interpreter value for attribute " << attr;
    return {EvalStatus::Completed, std::nullopt};
  }
  if (auto value = evaluable.toEvalValue(allocator))
    return {EvalStatus::Completed, std::move(value)};
  return {EvalStatus::Completed, std::nullopt};
}

} // namespace

Answer Engine::query(Value value) {
  assert(!queryAllocator &&
         "cannot start a query while another query is active");
  remaining = budget;
  cache.beginQuery();
  answerAllocator.beginRetaining();
  queryAllocator.emplace(ctx, false);
  Answer answer = evaluate(value);
  if (const auto &value = answer.getValue())
    answer.value = answerAllocator.retain(*value);
  queryAllocator.reset();
  return answer;
}

Answer Engine::queryNested(Value value) {
  assert(queryAllocator && "nested query requires an active query");
  return evaluate(value);
}

Answer Engine::evaluate(Value value) {
  Answer unknown{EvalStatus::Completed, std::nullopt};
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return unknown;

  Operation *op = result.getOwner();
  auto evaluable = dyn_cast<EvaluableOpInterface>(op);
  if (!evaluable && !op->hasTrait<OpTrait::ConstantLike>()) {
    LDBG() << "no evaluation function for '" << op->getName() << "'";
    return unknown;
  }

  // Cache-key methods and graph-region operands can re-enter an operation
  // whose evaluation is in progress.
  if (!active.insert(op).second)
    return {EvalStatus::Cycle, std::nullopt};
  llvm::scope_exit leave([&] { active.erase(op); });

  EvalScope scope(*this);
  std::optional<CacheLookup> cached;
  if (evaluable && evaluable.isCacheable()) {
    auto lookup = cache.lookup(op, scope);
    if (lookup.status != EvalStatus::Completed)
      return {lookup.status, std::nullopt};
    assert(lookup.getValue() && "completed lookup must provide a key");
    cached = *lookup.getValue();
    if (cached->entry)
      if (auto &answer = cached->entry->results[result.getResultNumber()])
        return *answer;
  }

  SmallVector<std::optional<EvalValue>> operands;
  for (Value operand : op->getOperands()) {
    Answer answer = evaluate(operand);
    if (answer.status != EvalStatus::Completed)
      return answer;
    operands.push_back(std::move(answer.value));
  }

  if (remaining == 0)
    return {EvalStatus::Exhausted, std::nullopt};
  --remaining;

  SmallVector<Answer> results;
  if (!evaluable) {
    results.push_back(evaluateConstant(op, *queryAllocator));
  } else {
    results = evaluable.evaluate(operands, scope);
  }

  assert(results.size() == op->getNumResults() &&
         "evaluation must return one answer per operation result");
  Answer answer = results[result.getResultNumber()];
  if (cached && llvm::any_of(results, [](const Answer &result) {
        return result.status == EvalStatus::Completed;
      })) {
    CacheEntry *entry = cached->entry;
    if (entry)
      cache.fill(entry, results);
    else
      entry = cache.insert(op, *cached, results, scope);
    if (auto &slot = entry->results[result.getResultNumber()])
      return *slot;
  }
  return answer;
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
