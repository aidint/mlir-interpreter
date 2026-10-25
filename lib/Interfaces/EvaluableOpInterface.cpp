#include "interpreter/Interfaces/EvaluableOpInterface.h"
#include "interpreter/EvalOperationEquivalence.h"

namespace mlir::interpreter {

EvalResult<llvm::hash_code> hashEvaluatedValue(Value value, EvalScope &scope) {
  Answer answer = scope.query(value);
  if (!answer.completed())
    return answer.status;
  if (const auto &known = answer.getValue())
    return known->hash();
  return mlir::hash_value(value);
}

EvalResult<bool> isEvaluatedValueEqual(Value lhs, Value rhs, EvalScope &scope) {
  if (lhs == rhs)
    return true;
  Answer lhsAnswer = scope.query(lhs);
  if (!lhsAnswer.completed())
    return lhsAnswer.status;
  const auto &lhsValue = lhsAnswer.getValue();
  if (!lhsValue)
    return false;
  Answer rhsAnswer = scope.query(rhs);
  if (!rhsAnswer.completed())
    return rhsAnswer.status;
  const auto &rhsValue = rhsAnswer.getValue();
  return rhsValue && lhsValue->isEqual(*rhsValue);
}

EvalResult<llvm::hash_code> detail::getDefaultHash(Operation *op,
                                                   EvalScope &scope) {
  return EvalOperationEquivalence::computeHash(
      op, [&](Value value) { return hashEvaluatedValue(value, scope); });
}

EvalResult<bool> detail::isDefaultEquivalent(Operation *op, Operation *other,
                                             EvalScope &scope) {
  return EvalOperationEquivalence::isEquivalentTo(
      op, other, [&](Value lhs, Value rhs) {
        return isEvaluatedValueEqual(lhs, rhs, scope);
      });
}

} // namespace mlir::interpreter

#include "interpreter/Interfaces/EvaluableOpInterface.cpp.inc"
