#ifndef INTERPRETER_EVALUABLEATTRINTERFACE
#define INTERPRETER_EVALUABLEATTRINTERFACE

include "mlir/IR/Interfaces.td"

def EvaluableAttrInterface : AttrInterface<"EvaluableAttrInterface"> {
  let description = [{
    Converts an attribute's contents into an interpreter value.
  }];
  let cppNamespace = "::mlir::interpreter";

  let methods = [
    InterfaceMethod<[{
        Returns the interpreter value this attribute denotes, or null when the
        model has no value for this instance, e.g. a float type it doesn't
        support.
      }],
      "::mlir::interpreter::EvalValue", "toEvalValue",
      (ins "::mlir::interpreter::EvalValueStorageAllocator &":$allocator)>,
    InterfaceMethod<[{
        Returns whether this attribute's value is kept across queries. Most
        values are cheaper to convert again than to cache, so the default is
        false; an attribute whose conversion copies its contents, e.g. a large
        `DenseElementsAttr`, may opt in.
      }],
      "bool", "isCacheable", (ins), [{}],
      /*defaultImplementation=*/[{
        return false;
      }]>
  ];
}

#endif
