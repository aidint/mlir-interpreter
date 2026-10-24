#ifndef INTERPRETER_EVALOPERATIONEQUIVALENCE_H
#define INTERPRETER_EVALOPERATIONEQUIVALENCE_H

#include "interpreter/EvalResult.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/Hashing.h"

namespace mlir::interpreter {

/// This class provides MLIR's `OperationEquivalence` hashing and comparison
/// with callbacks that may signal. Operand callbacks return `EvalResult`s, and
/// the first signal is returned immediately, so a callback can evaluate
/// operands without precomputing them. The utilities only traverse operations
/// and track which values correspond; the callbacks decide what an operand's
/// value is and how unknown values compare. Locations are always ignored, as
/// they don't change what an operation computes.
struct EvalOperationEquivalence {
  using HashOperandFn = function_ref<EvalResult<llvm::hash_code>(Value)>;
  using CheckEquivalentFn = function_ref<EvalResult<bool>(Value, Value)>;

  /// Hashes `op`'s name, attributes, properties and result types, and its
  /// operands through `hashOperand`. Operand hashes of a commutative op are
  /// summed, so their order doesn't matter. Results and regions are not hashed,
  /// as in MLIR's `computeHash`, so operations `isEquivalentTo` matches hash
  /// equally whenever `hashOperand` agrees with its `checkEquivalent`.
  static EvalResult<llvm::hash_code> computeHash(Operation *op,
                                                 HashOperandFn hashOperand);

  /// Compares `lhs` and `rhs` and their regions. Structure is matched first,
  /// without calling `checkEquivalent`: names, attributes, properties, result
  /// and block argument types, region shapes and successors. Values defined
  /// inside the compared operations, including their results, must then
  /// correspond, and every other pair of distinct operands of the same type,
  /// including values captured by a region, goes to `checkEquivalent`.
  /// Operands of a commutative op match in any order, with duplicates paired
  /// one to one.
  static EvalResult<bool> isEquivalentTo(Operation *lhs, Operation *rhs,
                                         CheckEquivalentFn checkEquivalent);
};

} // namespace mlir::interpreter

#endif
