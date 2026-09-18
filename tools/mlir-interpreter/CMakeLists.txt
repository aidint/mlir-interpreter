add_executable(mlir-interpreter mlir-interpreter.cpp)

target_link_libraries(mlir-interpreter PRIVATE
  MLIRInterpreter
  MLIRInterpreterArith
  MLIRInterpreterBuiltin
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRUBDialect
  LLVMSupport)
