#include "InterpTestDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::interp_test;

#include "InterpTestDialect.cpp.inc"

void InterpTestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "InterpTestOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "InterpTestOps.cpp.inc"
