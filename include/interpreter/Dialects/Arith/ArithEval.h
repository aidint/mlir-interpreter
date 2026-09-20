#ifndef INTERPRETER_DIALECTS_ARITH_ARITHEVAL_H
#define INTERPRETER_DIALECTS_ARITH_ARITHEVAL_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::interpreter {

void registerArithEvalExternalModels(DialectRegistry &registry);

} // namespace mlir::interpreter

#endif
