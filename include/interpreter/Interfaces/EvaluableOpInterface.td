#ifndef INTERPRETER_EVALUABLEOPINTERFACE
#define INTERPRETER_EVALUABLEOPINTERFACE

include "mlir/IR/OpBase.td"

def EvaluableOpInterface : OpInterface<"EvaluableOpInterface"> {
  let description = [{
    Computes an operation's results from its operands, each of which may be
    unknown.
  }];
  let cppNamespace = "::mlir::interpreter";

  let methods = [
    InterfaceMethod<
      "Computes a stable cache hash, or returns Exhausted/NeedsOrder. Defaults to the operation's identity.",
      "::mlir::interpreter::EvalResult<::llvm::hash_code>", "getHash",
      (ins "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return {::mlir::interpreter::EvalStatus::Completed,
                ::llvm::hash_value($_op.getOperation())};
      }]>,
    InterfaceMethod<
      "Compares cache keys symmetrically; equal operations must hash equally. May return Exhausted/NeedsOrder. Defaults to the operation's identity.",
      "::mlir::interpreter::EvalResult<bool>", "isEqual",
      (ins "::mlir::Operation *":$other,
           "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return {::mlir::interpreter::EvalStatus::Completed,
                $_op.getOperation() == other};
      }]>,
    InterfaceMethod<
      "Returns one answer per result.",
      "::llvm::SmallVector<::mlir::interpreter::Answer>", "evaluate",
      (ins "::llvm::ArrayRef<::std::optional<::mlir::interpreter::EvalValue>>":$operands,
           "::mlir::interpreter::EvalScope &":$scope)>,
    InterfaceMethod<
      "Returns whether stable results may be retained across queries.",
      "bool", "isCacheable", (ins), [{}],
      /*defaultImplementation=*/[{ return true; }]>
  ];
}

#endif
