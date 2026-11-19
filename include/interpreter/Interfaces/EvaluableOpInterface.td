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
        Returns one answer per result. Operands are queried through `scope`, and
        a signal from one must be propagated, e.g. with `Answer::propagate`.
      }],
      "::llvm::SmallVector<::mlir::interpreter::Answer>", "evaluate",
      (ins "::mlir::interpreter::EvalScope &":$scope)>,
    InterfaceMethod<[{
        Returns whether this operation's results may be retained across queries,
        or returns Exhausted, NeedsOrder or Cycle. A cached operation is keyed by
        its structure and the answers of its inputs, which are all evaluated
        before the lookup, so its results must depend on nothing else. By
        default, operations without side effects or successors are cached, as
        `defaultIsCacheable` decides.
      }],
      "::mlir::interpreter::EvalResult<bool>", "isCacheable",
      (ins "::mlir::interpreter::EvalScope &":$scope), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::interpreter::defaultIsCacheable($_op.getOperation());
      }]>
  ];
}

#endif
