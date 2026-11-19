#ifndef INTERP_TEST_OPS
#define INTERP_TEST_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/Properties.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def InterpTest_Dialect : Dialect {
  let name = "interp_test";
  let summary = "Operations for the interpreter's tests";
  let description = [{
    Operations with features that the dialects the tests link lack, such as a
    property that owns heap memory.
  }];
  let cppNamespace = "::mlir::interp_test";
}

def InterpTest_PropOp : Op<InterpTest_Dialect, "prop", [Pure]> {
  let summary = "An operation with a string property";
  let description = [{
    Produces an `i32`, and holds `label` as a `std::string` property. A label
    too long for the string's inline buffer owns heap memory, which only the
    property's destructor frees.
  }];
  let arguments = (ins StringProp:$label);
  let results = (outs I32:$result);
}

#endif
