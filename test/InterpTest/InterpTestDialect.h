#ifndef INTERPRETER_TEST_INTERPTEST_INTERPTESTDIALECT_H
#define INTERPRETER_TEST_INTERPTEST_INTERPTESTDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "InterpTestDialect.h.inc"

#define GET_OP_CLASSES
#include "InterpTestOps.h.inc"

#endif
