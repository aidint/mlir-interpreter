#ifndef MLIR_INTERPRETER_ARITHEVAL_H
#define MLIR_INTERPRETER_ARITHEVAL_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::interpreter {

void registerArithEvalExternalModels(DialectRegistry &registry);

} // namespace mlir::interpreter

#endif
