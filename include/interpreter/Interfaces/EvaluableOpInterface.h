#ifndef INTERPRETER_EVALUABLEOPINTERFACE_H
#define INTERPRETER_EVALUABLEOPINTERFACE_H

#include "interpreter/EvalResult.h"
#include "interpreter/EvalValueStorageAllocator.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <optional>

namespace mlir::interpreter {

class Engine;

/// The view an operation's evaluation gets of the engine that drives it.
class EvalScope {
public:
  explicit EvalScope(Engine &engine) : engine(engine) {}

  Answer query(Value value);
  SmallVector<Answer> walk(Region &region, ArrayRef<Answer> args);
  bool isOrdered() const;
  EvalValueStorageAllocator &getAllocator();

private:
  Engine &engine;
};

/// Returns `signal`'s status for every result of `op`, so an evaluation
/// function can propagate an operand's `Exhausted`, `NeedsOrder` or `Cycle`.
inline SmallVector<Answer> propagateSignal(Operation *op,
                                           const Answer &signal) {
  assert(signal.status != EvalStatus::Completed && "only signals propagate");
  return SmallVector<Answer>(op->getNumResults(),
                             Answer{signal.status, std::nullopt});
}

} // namespace mlir::interpreter

#include "interpreter/Interfaces/EvaluableOpInterface.h.inc"

#endif
