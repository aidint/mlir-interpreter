#ifndef MLIR_INTERPRETER_EVALUABLEATTRINTERFACE
#define MLIR_INTERPRETER_EVALUABLEATTRINTERFACE

include "mlir/IR/Interfaces.td"

def EvaluableAttrInterface : AttrInterface<"EvaluableAttrInterface"> {
  let description = [{
    Converts an attribute's contents into an interpreter value.
  }];
  let cppNamespace = "::mlir::interpreter";

  let methods = [
    InterfaceMethod<
      "Returns the interpreter value this attribute denotes, if any.",
      "::std::optional<::mlir::interpreter::InterpreterValue>",
      "toInterpreterValue", (ins)>
  ];
}

#endif
