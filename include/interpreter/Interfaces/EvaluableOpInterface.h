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

#include "llvm/ADT/SmallVector.h"

namespace mlir::interpreter {

class Engine;

/// This class represents the arguments of a block bound for one loop
/// iteration. Destroying it unbinds them and drops the answers memoized in the
/// iteration.
class [[nodiscard]] BlockBinding {
public:
  BlockBinding(const BlockBinding &) = delete;
  BlockBinding &operator=(const BlockBinding &) = delete;
  ~BlockBinding();

private:
  friend class EvalScope;

  BlockBinding(Engine &engine, Block *block) : engine(engine), block(block) {}

  Engine &engine;
  Block *block;
};

/// This class represents a freeze on the cache: while any freeze lives, the
/// cache is still looked up, but no answer is stored.
class [[nodiscard]] CacheFreeze {
public:
  CacheFreeze(const CacheFreeze &) = delete;
  CacheFreeze &operator=(const CacheFreeze &) = delete;
  ~CacheFreeze();

private:
  friend class EvalScope;

  explicit CacheFreeze(Engine &engine);

  Engine &engine;
};

/// The view an operation's evaluation gets of the engine that drives it.
class EvalScope {
public:
  explicit EvalScope(Engine &engine) : engine(engine) {}

  Answer query(Value value);
  /// Freezes the cache until the returned freeze is destroyed. Freezes nest.
  CacheFreeze freezeCache();
  /// Spends a step of the query's budget on work the operation does itself,
  /// such as a loop iteration, or returns false when the budget ran out.
  [[nodiscard]] bool declareCost();
  /// Binds `block`'s arguments to `args` until the returned binding is
  /// destroyed. Bindings nest, and must be destroyed in reverse order.
  BlockBinding bind(Block *block, ArrayRef<Answer> args);
  SmallVector<Answer> walk(Region &region, ArrayRef<Answer> args);
  bool isOrdered() const;
  EvalValueStorageAllocator &getAllocator();

private:
  Engine &engine;
};

/// The default `isCacheable`: whether `op` has no side effects and no
/// successors. An override can call it to keep these rules.
bool defaultIsCacheable(Operation *op);

} // namespace mlir::interpreter

#include "interpreter/Interfaces/EvaluableOpInterface.h.inc"

#endif
