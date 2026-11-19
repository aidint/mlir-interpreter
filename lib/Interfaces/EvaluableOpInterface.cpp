#include "interpreter/Interfaces/EvaluableOpInterface.h"

namespace mlir::interpreter {

bool defaultIsCacheable(Operation *op) {
  // An op with side effects can compute different results from equal
  // operands, and a cache key can't hold the blocks that successors name.
  return isMemoryEffectFree(op) && op->getNumSuccessors() == 0;
}

} // namespace mlir::interpreter

#include "interpreter/Interfaces/EvaluableOpInterface.cpp.inc"
