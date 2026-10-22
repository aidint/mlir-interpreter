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
      "Computes a stable cache hash, or returns Exhausted, NeedsOrder or Cycle. Defaults to `EvalOperationEquivalence::computeHash` over the evaluated operands; an unknown operand hashes by its SSA value.",
      "::mlir::interpreter::EvalResult<::llvm::hash_code>", "getHash",
      (ins "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::interpreter::detail::getDefaultHash($_op.getOperation(),
                                                         scope);
      }]>,
    InterfaceMethod<
      "Compares cache keys symmetrically; equivalent operations must hash equally. May return Exhausted, NeedsOrder or Cycle. Defaults to `EvalOperationEquivalence::isEquivalentTo`, which compares structure and regions, then values from outside the operations by their evaluated values; an unknown value only equals the same SSA value.",
      "::mlir::interpreter::EvalResult<bool>", "isEquivalentTo",
      (ins "::mlir::Operation *":$other,
           "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::interpreter::detail::isDefaultEquivalent(
            $_op.getOperation(), other, scope);
      }]>,
    InterfaceMethod<
      "Returns one answer per result. Operands are queried through `scope`, and a signal from one must be propagated, e.g. with `Answer::propagate`.",
      "::llvm::SmallVector<::mlir::interpreter::Answer>", "evaluate",
      (ins "::mlir::interpreter::EvalScope &":$scope)>,
    InterfaceMethod<
      "Returns whether this operation's results may be retained across queries, or returns Exhausted, NeedsOrder or Cycle. It may query operands through `scope`. `getHash` and `isEquivalentTo` stay callable when it is false. Defaults to true.",
      "::mlir::interpreter::EvalResult<bool>", "isCacheable",
      (ins "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return {::mlir::interpreter::EvalStatus::Completed, true};
      }]>
  ];
}

#endif
