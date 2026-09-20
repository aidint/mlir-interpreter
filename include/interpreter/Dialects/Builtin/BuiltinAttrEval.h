#ifndef INTERPRETER_DIALECTS_BUILTIN_BUILTINATTREVAL_H
#define INTERPRETER_DIALECTS_BUILTIN_BUILTINATTREVAL_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::interpreter {

void registerBuiltinAttrEvalExternalModels(DialectRegistry &registry);

} // namespace mlir::interpreter

#endif
