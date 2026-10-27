#ifndef INTERPRETER_EVALUABLEOPINTERFACE_H
#define INTERPRETER_EVALUABLEOPINTERFACE_H

#include "interpreter/Answer.h"
#include "interpreter/EvalResult.h"
#include "interpreter/EvalValueStorageAllocator.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::interpreter {

class Engine;

/// This class represents the arguments of a block bound for one loop
/// iteration. Destroying it unbinds them and drops the answers memoized in the
/// iteration. It is empty, and binds nothing, when the budget ran out.
class [[nodiscard]] BlockBinding {
public:
  BlockBinding(const BlockBinding &) = delete;
  BlockBinding &operator=(const BlockBinding &) = delete;
  ~BlockBinding();

  explicit operator bool() const { return block; }

private:
  friend class EvalScope;

  BlockBinding(Engine &engine, Block *block) : engine(engine), block(block) {}

  Engine &engine;
  Block *block;
};

/// The view an operation's evaluation gets of the engine that drives it.
class EvalScope {
public:
  explicit EvalScope(Engine &engine) : engine(engine) {}

  Answer query(Value value);
  /// Binds `block`'s arguments to `args` until the returned binding is
  /// destroyed. Bindings nest, and must be destroyed in reverse order. Each
  /// binding costs a step, so a loop whose body costs nothing still ends.
  BlockBinding bind(Block *block, ArrayRef<Answer> args);
  SmallVector<Answer> walk(Region &region, ArrayRef<Answer> args);
  bool isOrdered() const;
  EvalValueStorageAllocator &getAllocator();

private:
  Engine &engine;
};

/// Hashes `value` by its evaluated value, or by its SSA identity when it is
/// unknown, consistently with `isEvaluatedValueEqual`. A signal from
/// evaluating it is returned.
EvalResult<llvm::hash_code> hashEvaluatedValue(Value value, EvalScope &scope);

/// Returns whether `lhs` and `rhs` evaluate to equal values. An unknown value
/// only equals itself, so `rhs` is not queried when `lhs` is unknown. A signal
/// from evaluating either is returned.
EvalResult<bool> isEvaluatedValueEqual(Value lhs, Value rhs, EvalScope &scope);

namespace detail {
/// The default `getHash` and `isEquivalentTo`: `EvalOperationEquivalence`
/// with `hashEvaluatedValue` and `isEvaluatedValueEqual`.
EvalResult<llvm::hash_code> getDefaultHash(Operation *op, EvalScope &scope);
EvalResult<bool> isDefaultEquivalent(Operation *op, Operation *other,
                                     EvalScope &scope);
} // namespace detail

} // namespace mlir::interpreter

#include "interpreter/Interfaces/EvaluableOpInterface.h.inc"

#endif
