#include "mlir-interpreter/Engine.h"
#include "mlir-interpreter/EvaluableAttrInterface.h"

#include "mlir/IR/Matchers.h"

#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "mlir-interpreter"

namespace mlir::interpreter {

namespace {

class EngineEvalContext : public EvalContext {
public:
  explicit EngineEvalContext(Engine &engine) : engine(engine) {}

  Answer query(Value value) override { return engine.query(value); }

  SmallVector<Answer> walk(Region &region, ArrayRef<Answer> args) override {
    return {};
  }

  bool isOrdered() const override { return false; }

private:
  Engine &engine;
};

Answer evaluateConstant(Operation *op) {
  Attribute attr;
  if (!matchPattern(op, m_Constant(&attr)))
    return {AnswerKind::Unknown, std::nullopt};
  auto evaluable = dyn_cast<EvaluableAttrInterface>(attr);
  if (!evaluable) {
    LDBG() << "no interpreter value for attribute " << attr;
    return {AnswerKind::Unknown, std::nullopt};
  }
  if (auto value = evaluable.toInterpreterValue())
    return {AnswerKind::Known, std::move(value)};
  return {AnswerKind::Unknown, std::nullopt};
}

} // namespace

Engine::Engine(uint64_t budget) : budget(budget) {}

void Engine::registerType(TypeID type) {}

Answer Engine::query(Value value) {
  if (depth == 0)
    remaining = budget;
  ++depth;
  Answer answer = evaluate(value);
  --depth;
  return answer;
}

Answer Engine::evaluate(Value value) {
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

  SmallVector<std::optional<InterpreterValue>> operands;
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

  if (!evaluable)
    return evaluateConstant(op);

  EngineEvalContext ctx(*this);
  SmallVector<Answer> results = evaluable.evaluate(operands, ctx);
  return results[result.getResultNumber()];
}

FailureOr<func::FuncOp>
Engine::specialize(func::CallOp call,
                   ArrayRef<std::optional<InterpreterValue>> args) {
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
