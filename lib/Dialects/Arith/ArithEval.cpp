#include "interpreter/Dialects/Arith/ArithEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/APInt.h"

namespace mlir::interpreter {
namespace {

Answer known(EvalScope &scope, APInt value) {
  return IntEvalValue::get(scope.getAllocator(), std::move(value));
}

const APInt *getInt(const Answer &answer) {
  const auto &value = answer.getValue();
  return value ? &cast<IntEvalValue>(value).getValue() : nullptr;
}

bool isZero(const Answer &answer) {
  const APInt *value = getInt(answer);
  return value && value->isZero();
}

/// Evaluates an operation whose result is unknown when either operand is, so
/// an unknown left operand leaves the right one unqueried.
SmallVector<Answer>
evaluateKnown(Operation *op, EvalScope &scope,
              function_ref<APInt(const APInt &, const APInt &)> fn) {
  Answer lhs = scope.query(op->getOperand(0));
  if (!lhs.completed())
    return lhs.propagate(op);
  const APInt *lhsInt = getInt(lhs);
  if (!lhsInt)
    return {Answer::unknown()};
  Answer rhs = scope.query(op->getOperand(1));
  if (!rhs.completed())
    return rhs.propagate(op);
  const APInt *rhsInt = getInt(rhs);
  if (!rhsInt)
    return {Answer::unknown()};
  return {known(scope, fn(*lhsInt, *rhsInt))};
}

/// This class implements the evaluation of `arith.addi`.
struct AddIOpEval
    : EvaluableOpInterface::ExternalModel<AddIOpEval, arith::AddIOp> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    return evaluateKnown(op, scope, [](const APInt &lhs, const APInt &rhs) {
      return lhs + rhs;
    });
  }
};

/// This class implements the evaluation of `arith.subi`.
struct SubIOpEval
    : EvaluableOpInterface::ExternalModel<SubIOpEval, arith::SubIOp> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    return evaluateKnown(op, scope, [](const APInt &lhs, const APInt &rhs) {
      return lhs - rhs;
    });
  }
};

/// This class implements the evaluation of `arith.muli`. A zero operand
/// decides the product, so it isn't cached when its left operand is zero.
struct MulIOpEval
    : EvaluableOpInterface::ExternalModel<MulIOpEval, arith::MulIOp> {
  EvalResult<bool> isCacheable(Operation *op, EvalScope &scope) const {
    if (!defaultIsCacheable(op))
      return false;
    // A zero left operand decides the product without the right one, which a
    // cache lookup would evaluate.
    Answer lhs = scope.query(op->getOperand(0));
    if (!lhs.completed())
      return lhs.status;
    return !isZero(lhs);
  }

  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    // A zero operand decides the product, so the other one is never queried.
    Answer lhs = scope.query(op->getOperand(0));
    if (!lhs.completed())
      return lhs.propagate(op);
    const APInt *lhsInt = getInt(lhs);
    if (lhsInt && lhsInt->isZero())
      return {lhs};
    Answer rhs = scope.query(op->getOperand(1));
    if (!rhs.completed())
      return rhs.propagate(op);
    const APInt *rhsInt = getInt(rhs);
    if (rhsInt && rhsInt->isZero())
      return {rhs};
    if (!lhsInt || !rhsInt)
      return {Answer::unknown()};
    return {known(scope, *lhsInt * *rhsInt)};
  }
};

} // namespace

void registerArithEvalExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *) {
    arith::AddIOp::attachInterface<AddIOpEval>(*ctx);
    arith::MulIOp::attachInterface<MulIOpEval>(*ctx);
    arith::SubIOp::attachInterface<SubIOpEval>(*ctx);
  });
}

} // namespace mlir::interpreter
