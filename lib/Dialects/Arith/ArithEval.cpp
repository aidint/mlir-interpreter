#include "interpreter/Dialects/Arith/ArithEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Interfaces/EquatableEvalValueInterface.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"

#include <algorithm>

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
  return {known(scope, fn(*lhsInt, *rhsInt))};
}

arith::IntegerOverflowFlags getOverflowFlags(Operation *op) {
  if (auto attr =
          cast<arith::ArithIntegerOverflowFlagsInterface>(op).getOverflowAttr())
    return attr.getValue();
  return arith::IntegerOverflowFlags::none;
}

/// This class represents the operand part of an integer binary operation's
/// cache key. Known operands are keyed by value, a product with a known zero
/// operand by its zero alone, and any other unknown operand by the operation
/// itself.
struct BinaryKey {
  /// The operation, when the key falls back to its identity.
  Operation *identity = nullptr;
  bool zeroProduct = false;
  EvalValue lhs, rhs;
};

/// Queries an operand for a key, dropping a value that can't be compared.
Answer queryKeyOperand(Operation *op, unsigned index, EvalScope &scope) {
  Answer answer = scope.query(op->getOperand(index));
  if (const auto &value = answer.getValue())
    if (!isa<EquatableEvalValueInterface>(*value))
      answer.value.reset();
  return answer;
}

/// Returns `op`'s key, querying only the operands its evaluation queries.
/// Operand signals propagate instead of changing the key.
EvalResult<BinaryKey> getKey(Operation *op, EvalScope &scope) {
  bool product = isa<arith::MulIOp>(op);
  BinaryKey identity{op};
  BinaryKey zero{nullptr, /*zeroProduct=*/true};

  Answer lhs = queryKeyOperand(op, 0, scope);
  if (lhs.status != EvalStatus::Completed)
    return {lhs.status, std::nullopt};
  if (product && isZero(lhs))
    return {EvalStatus::Completed, zero};
  // A product still queries its right operand, which may be zero.
  if (!lhs.getValue() && !product)
    return {EvalStatus::Completed, identity};

  Answer rhs = queryKeyOperand(op, 1, scope);
  if (rhs.status != EvalStatus::Completed)
    return {rhs.status, std::nullopt};
  if (product && isZero(rhs))
    return {EvalStatus::Completed, zero};
  if (!lhs.getValue() || !rhs.getValue())
    return {EvalStatus::Completed, identity};
  return {EvalStatus::Completed,
          BinaryKey{nullptr, false, *lhs.getValue(), *rhs.getValue()}};
}

/// This class implements the semantic cache key of `arith.addi`,
/// `arith.muli` and `arith.subi`: the operation kind, result type and
/// overflow flags, plus a `BinaryKey`. Operands of commutative operations
/// match in either order.
template <typename ConcreteModel, typename OpT>
struct BinaryOpEval : EvaluableOpInterface::ExternalModel<ConcreteModel, OpT> {
  EvalResult<llvm::hash_code> getHash(Operation *op, EvalScope &scope) const {
    auto key = getKey(op, scope);
    if (key.status != EvalStatus::Completed)
      return {key.status, std::nullopt};
    const BinaryKey &k = *key.getValue();
    if (k.identity)
      return {EvalStatus::Completed, llvm::hash_value(k.identity)};
    llvm::hash_code head =
        llvm::hash_combine(op->getName(), op->getResult(0).getType(),
                           getOverflowFlags(op), k.zeroProduct);
    if (k.zeroProduct)
      return {EvalStatus::Completed, head};
    size_t lhs = cast<EquatableEvalValueInterface>(k.lhs).hash();
    size_t rhs = cast<EquatableEvalValueInterface>(k.rhs).hash();
    if (op->hasTrait<OpTrait::IsCommutative>() && rhs < lhs)
      std::swap(lhs, rhs);
    return {EvalStatus::Completed, llvm::hash_combine(head, lhs, rhs)};
  }

  EvalResult<bool> isEqual(Operation *op, Operation *other,
                           EvalScope &scope) const {
    if (op->getName() != other->getName() ||
        op->getResultTypes() != other->getResultTypes() ||
        getOverflowFlags(op) != getOverflowFlags(other))
      return {EvalStatus::Completed, false};
    auto key = getKey(op, scope);
    if (key.status != EvalStatus::Completed)
      return {key.status, std::nullopt};
    auto otherKey = getKey(other, scope);
    if (otherKey.status != EvalStatus::Completed)
      return {otherKey.status, std::nullopt};
    const BinaryKey &a = *key.getValue();
    const BinaryKey &b = *otherKey.getValue();
    if (a.identity || b.identity)
      return {EvalStatus::Completed, a.identity == b.identity};
    if (a.zeroProduct || b.zeroProduct)
      return {EvalStatus::Completed, a.zeroProduct == b.zeroProduct};
    bool equal = interpreter::isEqual(a.lhs, b.lhs) &&
                 interpreter::isEqual(a.rhs, b.rhs);
    if (!equal && op->hasTrait<OpTrait::IsCommutative>())
      equal = interpreter::isEqual(a.lhs, b.rhs) &&
              interpreter::isEqual(a.rhs, b.lhs);
    return {EvalStatus::Completed, equal};
  }
};

struct AddIOpEval : BinaryOpEval<AddIOpEval, arith::AddIOp> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    return evaluateKnown(op, scope, [](const APInt &lhs, const APInt &rhs) {
      return lhs + rhs;
    });
  }
};

struct SubIOpEval : BinaryOpEval<SubIOpEval, arith::SubIOp> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    return evaluateKnown(op, scope, [](const APInt &lhs, const APInt &rhs) {
      return lhs - rhs;
    });
  }
};

struct MulIOpEval : BinaryOpEval<MulIOpEval, arith::MulIOp> {
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
    arith::SubIOp::attachInterface<SubIOpEval>(*ctx);
  });
}

} // namespace mlir::interpreter
