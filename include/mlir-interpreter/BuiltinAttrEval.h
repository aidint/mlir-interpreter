#ifndef MLIR_INTERPRETER_BUILTINATTREVAL_H
#define MLIR_INTERPRETER_BUILTINATTREVAL_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::interpreter {

void registerBuiltinAttrEvalExternalModels(DialectRegistry &registry);

} // namespace mlir::interpreter

#endif
