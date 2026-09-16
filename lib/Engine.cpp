#include "mlir-interpreter/Engine.h"

namespace mlir::interpreter {

EvalContext::EvalContext(Engine &engine) : engine(engine) {}

Answer EvalContext::query(Value value) { return engine.query(value); }

SmallVector<Answer> EvalContext::walk(Region &region, ArrayRef<Answer> args) {
  return {};
}

bool EvalContext::isOrdered() const { return false; }

Engine::Engine(uint64_t budget) : budget(budget) {}

void Engine::registerOp(StringRef opName, EvalFn fn) {}

void Engine::registerType(TypeID type) {}

Answer Engine::query(Value value) {
  return {AnswerKind::Unknown, std::nullopt};
}

FailureOr<func::FuncOp>
Engine::specialize(func::CallOp call,
                   ArrayRef<std::optional<InterpValue>> args) {
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
