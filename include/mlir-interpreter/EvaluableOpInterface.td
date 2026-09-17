#ifndef MLIR_INTERPRETER_EVALUABLEOPINTERFACE
#define MLIR_INTERPRETER_EVALUABLEOPINTERFACE

include "mlir/IR/OpBase.td"

def EvaluableOpInterface : OpInterface<"EvaluableOpInterface"> {
  let description = [{
    Computes an operation's results from its operands, each of which may be
    unknown.
  }];
  let cppNamespace = "::mlir::interpreter";

  let methods = [
    InterfaceMethod<
      "Returns one answer per result.",
      "::llvm::SmallVector<::mlir::interpreter::Answer>", "evaluate",
      (ins "::llvm::ArrayRef<::std::optional<::mlir::interpreter::InterpreterValue>>":$operands,
           "::mlir::interpreter::EvalContext &":$ctx)>
  ];
}

#endif
