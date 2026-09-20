add_executable(interpreter interpreter.cpp)

target_link_libraries(interpreter PRIVATE
  Interpreter
  InterpreterArith
  InterpreterBuiltin
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
