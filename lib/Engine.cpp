#include "interpreter/Engine.h"
#include "interpreter/EvalCache.h"
#include "interpreter/Interfaces/EvaluableAttrInterface.h"

#include "mlir/IR/Matchers.h"

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
    return {AnswerKind::Unknown, std::nullopt};
  auto evaluable = dyn_cast<EvaluableAttrInterface>(attr);
  if (!evaluable) {
    LDBG() << "no interpreter value for attribute " << attr;
    return {AnswerKind::Unknown, std::nullopt};
  }
  if (auto value = evaluable.toEvalValue(allocator))
    return {AnswerKind::Known, std::move(value)};
  return {AnswerKind::Unknown, std::nullopt};
}

} // namespace

Engine::Engine(EvalContext &ctx, uint64_t budget)
    : ctx(ctx), budget(budget), answerAllocator(ctx, false) {}

Answer Engine::query(Value value) {
  assert(!queryAllocator &&
         "cannot start a query while another query is active");
  remaining = budget;
  ctx.getCache().beginQuery();
  answerAllocator.beginRetaining();
  queryAllocator.emplace(ctx, false);
  Answer answer = evaluate(value);
  if (answer.value)
    answer.value = answerAllocator.retain(*answer.value);
  queryAllocator.reset();
  return answer;
}

Answer Engine::queryNested(Value value) {
  assert(queryAllocator && "nested query requires an active query");
  return evaluate(value);
}

Answer Engine::evaluate(Value value) {
  std::optional<EvalValue> cached;
  if (ctx.getCache().lookup(value, cached))
    return {cached ? AnswerKind::Known : AnswerKind::Unknown, cached};

  Answer unknown{AnswerKind::Unknown, std::nullopt};
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return unknown;

  Operation *op = result.getOwner();
  auto evaluable = dyn_cast<EvaluableOpInterface>(op);
  if (!evaluable && !op->hasTrait<OpTrait::ConstantLike>()) {
    LDBG() << "no evaluation function for '" << op->getName() << "'";
    return unknown;
  }

  SmallVector<std::optional<EvalValue>> operands;
  for (Value operand : op->getOperands()) {
    Answer answer = evaluate(operand);
    if (answer.kind == AnswerKind::Exhausted ||
        answer.kind == AnswerKind::NeedsOrder)
      return answer;
    operands.push_back(std::move(answer.value));
  }

  if (remaining == 0)
    return {AnswerKind::Exhausted, std::nullopt};
  --remaining;

  bool cacheable = !evaluable || evaluable.isCacheable();
  SmallVector<Answer> results;
  if (!evaluable) {
    results.push_back(evaluateConstant(op, *queryAllocator));
  } else {
    EvalScope scope(*this);
    results = evaluable.evaluate(operands, scope);
  }

  assert(results.size() == op->getNumResults() &&
         "evaluation must return one answer per operation result");
  if (cacheable) {
    for (auto [opResult, answer] : llvm::zip(op->getResults(), results)) {
      if (answer.kind == AnswerKind::Known ||
          answer.kind == AnswerKind::Unknown)
        ctx.getCache().insert(opResult, answer.value);
    }
  }

  Answer answer = results[result.getResultNumber()];
  if (cacheable && (answer.kind == AnswerKind::Known ||
                    answer.kind == AnswerKind::Unknown)) {
    bool found = ctx.getCache().lookup(value, answer.value);
    assert(found && "cacheable result was not inserted");
  }
  return answer;
}

FailureOr<func::FuncOp>
Engine::specialize(func::CallOp call, ArrayRef<std::optional<EvalValue>> args) {
  return failure();
}

StringRef stringifyAnswerKind(AnswerKind kind) {
  switch (kind) {
  case AnswerKind::Known:
    return "known";
  case AnswerKind::Unknown:
    return "unknown";
  case AnswerKind::Exhausted:
    return "exhausted";
  case AnswerKind::NeedsOrder:
    return "needs order";
  }
  llvm_unreachable("unhandled AnswerKind");
}

} // namespace mlir::interpreter
