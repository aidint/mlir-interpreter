#ifndef INTERPRETER_DIALECTS_SCF_SCFEVAL_H
#define INTERPRETER_DIALECTS_SCF_SCFEVAL_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::interpreter {

void registerSCFEvalExternalModels(DialectRegistry &registry);

} // namespace mlir::interpreter

#endif
