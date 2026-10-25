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
    InterfaceMethod<[{
        Hashes this operation, or returns Exhausted, NeedsOrder or Cycle.
        Operations that `isEquivalentTo` matches must hash equally.
      }],
      "::mlir::interpreter::EvalResult<::llvm::hash_code>", "getHash",
      (ins "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::interpreter::detail::getDefaultHash($_op.getOperation(),
                                                         scope);
      }]>,
    InterfaceMethod<[{
        Returns whether `other` computes the same results as this operation, or
        returns Exhausted, NeedsOrder or Cycle. Must be symmetric.
      }],
      "::mlir::interpreter::EvalResult<bool>", "isEquivalentTo",
      (ins "::mlir::Operation *":$other,
           "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::interpreter::detail::isDefaultEquivalent(
            $_op.getOperation(), other, scope);
      }]>,
    InterfaceMethod<[{
        Returns one answer per result. Operands are queried through `scope`, and
        a signal from one must be propagated, e.g. with `Answer::propagate`.
      }],
      "::llvm::SmallVector<::mlir::interpreter::Answer>", "evaluate",
      (ins "::mlir::interpreter::EvalScope &":$scope)>,
    InterfaceMethod<[{
        Returns whether this operation's results may be retained across queries,
        or returns Exhausted, NeedsOrder or Cycle.
      }],
      "::mlir::interpreter::EvalResult<bool>", "isCacheable",
      (ins "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return {::mlir::interpreter::EvalStatus::Completed, true};
      }]>
  ];
}

#endif
