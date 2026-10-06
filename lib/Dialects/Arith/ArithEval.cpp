#include "interpreter/Dialects/Arith/ArithEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/APInt.h"

namespace mlir::interpreter {
namespace {

Answer known(EvalScope &scope, APInt value) {
  return {EvalStatus::Completed,
          IntEvalValue::get(scope.getAllocator(), std::move(value))};
}

Answer unknown() { return {EvalStatus::Completed, std::nullopt}; }

const APInt *getInt(const Answer &answer) {
  const auto &value = answer.getValue();
  return value ? &cast<IntEvalValue>(*value).getValue() : nullptr;
}

struct AddIOpEval
    : EvaluableOpInterface::ExternalModel<AddIOpEval, arith::AddIOp> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    // An unknown operand makes the sum unknown, so the other one is never
    // queried.
    Answer lhs = scope.query(op->getOperand(0));
    if (lhs.status != EvalStatus::Completed)
      return propagateSignal(op, lhs);
    const APInt *lhsInt = getInt(lhs);
    if (!lhsInt)
      return {unknown()};
    Answer rhs = scope.query(op->getOperand(1));
    if (rhs.status != EvalStatus::Completed)
      return propagateSignal(op, rhs);
    const APInt *rhsInt = getInt(rhs);
    if (!rhsInt)
      return {unknown()};
    return {known(scope, *lhsInt + *rhsInt)};
  }
};

struct MulIOpEval
    : EvaluableOpInterface::ExternalModel<MulIOpEval, arith::MulIOp> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    // A zero operand decides the product, so the other one is never queried.
    Answer lhs = scope.query(op->getOperand(0));
    if (lhs.status != EvalStatus::Completed)
      return propagateSignal(op, lhs);
    const APInt *lhsInt = getInt(lhs);
    if (lhsInt && lhsInt->isZero())
      return {lhs};
    Answer rhs = scope.query(op->getOperand(1));
    if (rhs.status != EvalStatus::Completed)
      return propagateSignal(op, rhs);
    const APInt *rhsInt = getInt(rhs);
    if (rhsInt && rhsInt->isZero())
      return {rhs};
    if (!lhsInt || !rhsInt)
      return {unknown()};
    return {known(scope, *lhsInt * *rhsInt)};
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
