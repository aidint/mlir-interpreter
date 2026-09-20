#include "interpreter/BuiltinEvalValues.h"

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::interpreter::IntEvalValue)

namespace mlir::interpreter {

void registerBuiltinEvalValues(EvalContext &ctx) {
  ctx.registerEvalValue<IntEvalValue>();
}

} // namespace mlir::interpreter
