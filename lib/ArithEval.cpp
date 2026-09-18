#include "mlir-interpreter/ArithEval.h"
#include "mlir-interpreter/EvaluableOpInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/APInt.h"

namespace mlir::interpreter {
namespace {

Answer known(APInt value) {
  return {AnswerKind::Known, InterpreterValue{std::move(value)}};
}

Answer unknown() { return {AnswerKind::Unknown, std::nullopt}; }

const APInt *getInt(const std::optional<InterpreterValue> &value) {
  return value ? std::any_cast<APInt>(&value->payload) : nullptr;
}

struct AddIOpEval
    : EvaluableOpInterface::ExternalModel<AddIOpEval, arith::AddIOp> {
  SmallVector<Answer> evaluate(Operation *,
                               ArrayRef<std::optional<InterpreterValue>> operands,
                               EvalContext &) const {
    const APInt *lhs = getInt(operands[0]);
    const APInt *rhs = getInt(operands[1]);
    if (!lhs || !rhs)
      return {unknown()};
    return {known(*lhs + *rhs)};
  }
};

struct MulIOpEval
    : EvaluableOpInterface::ExternalModel<MulIOpEval, arith::MulIOp> {
  SmallVector<Answer> evaluate(Operation *,
                               ArrayRef<std::optional<InterpreterValue>> operands,
                               EvalContext &) const {
    const APInt *lhs = getInt(operands[0]);
    const APInt *rhs = getInt(operands[1]);
    if (lhs && lhs->isZero())
      return {known(*lhs)};
    if (rhs && rhs->isZero())
      return {known(*rhs)};
    if (!lhs || !rhs)
      return {unknown()};
    return {known(*lhs * *rhs)};
  }
};

} // namespace

void registerArithEvalExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *) {
    arith::AddIOp::attachInterface<AddIOpEval>(*ctx);
    arith::MulIOp::attachInterface<MulIOpEval>(*ctx);
  });
}

} // namespace mlir::interpreter
