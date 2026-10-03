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

const APInt *getInt(const std::optional<EvalValue> &value) {
  return value ? &cast<IntEvalValue>(*value).getValue() : nullptr;
}

struct AddIOpEval
    : EvaluableOpInterface::ExternalModel<AddIOpEval, arith::AddIOp> {
  EvalResult<llvm::hash_code> getHash(Operation *op, Engine &) const {
    return {EvalStatus::Completed, llvm::hash_value(op)};
  }

  EvalResult<bool> isEqual(Operation *op, Operation *other, Engine &) const {
    return {EvalStatus::Completed, op == other};
  }

  SmallVector<Answer> evaluate(Operation *,
                               ArrayRef<std::optional<EvalValue>> operands,
                               EvalScope &scope) const {
    const APInt *lhs = getInt(operands[0]);
    const APInt *rhs = getInt(operands[1]);
    if (!lhs || !rhs)
      return {unknown()};
    return {known(scope, *lhs + *rhs)};
  }
};

struct MulIOpEval
    : EvaluableOpInterface::ExternalModel<MulIOpEval, arith::MulIOp> {
  EvalResult<llvm::hash_code> getHash(Operation *op, Engine &) const {
    return {EvalStatus::Completed, llvm::hash_value(op)};
  }

  EvalResult<bool> isEqual(Operation *op, Operation *other, Engine &) const {
    return {EvalStatus::Completed, op == other};
  }

  SmallVector<Answer> evaluate(Operation *,
                               ArrayRef<std::optional<EvalValue>> operands,
                               EvalScope &scope) const {
    const APInt *lhs = getInt(operands[0]);
    const APInt *rhs = getInt(operands[1]);
    if (lhs && lhs->isZero())
      return {known(scope, *lhs)};
    if (rhs && rhs->isZero())
      return {known(scope, *rhs)};
    if (!lhs || !rhs)
      return {unknown()};
    return {known(scope, *lhs * *rhs)};
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
