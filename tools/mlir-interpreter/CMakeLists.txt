add_executable(mlir-interpreter mlir-interpreter.cpp)

target_link_libraries(mlir-interpreter PRIVATE
  MLIRInterpreter
  MLIRInterpreterArith
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRUBDialect
  LLVMSupport)
