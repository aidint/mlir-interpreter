#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::interpreter::IntEvalValue)

namespace mlir::interpreter {

void registerBuiltinEvalValues(EvalContext &ctx) {
  ctx.registerEvalValues<IntEvalValue>();
}

} // namespace mlir::interpreter
