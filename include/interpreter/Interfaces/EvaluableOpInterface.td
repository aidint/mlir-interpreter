#ifndef INTERPRETER_EVALUABLEOPINTERFACE
#define INTERPRETER_EVALUABLEOPINTERFACE

include "mlir/IR/OpBase.td"

def EvaluableOpInterface : OpInterface<"EvaluableOpInterface"> {
  let description = [{
    Evaluates an operation's results, querying through `EvalScope` only the
    operands it needs.
  }];
  let cppNamespace = "::mlir::interpreter";

  let methods = [
    InterfaceMethod<
      "Computes a stable cache hash, or returns Exhausted, NeedsOrder or Cycle. Defaults to the operation's identity.",
      "::mlir::interpreter::EvalResult<::llvm::hash_code>", "getHash",
      (ins "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return {::mlir::interpreter::EvalStatus::Completed,
                ::llvm::hash_value($_op.getOperation())};
      }]>,
    InterfaceMethod<
      "Compares cache keys symmetrically; equal operations must hash equally. May return Exhausted, NeedsOrder or Cycle. Defaults to the operation's identity.",
      "::mlir::interpreter::EvalResult<bool>", "isEqual",
      (ins "::mlir::Operation *":$other,
           "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return {::mlir::interpreter::EvalStatus::Completed,
                $_op.getOperation() == other};
      }]>,
    InterfaceMethod<
      "Returns one answer per result. Operands are queried through `scope`, and a signal from one must be propagated, e.g. with `propagateSignal`.",
      "::llvm::SmallVector<::mlir::interpreter::Answer>", "evaluate",
      (ins "::mlir::interpreter::EvalScope &":$scope)>,
    InterfaceMethod<
      "Returns whether stable results may be retained across queries.",
      "bool", "isCacheable", (ins), [{}],
      /*defaultImplementation=*/[{ return true; }]>
  ];
}

#endif
