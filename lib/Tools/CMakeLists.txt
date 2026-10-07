add_library(InterpreterRunner Runner.cpp)

target_link_libraries(InterpreterRunner PUBLIC
  Interpreter
  InterpreterArith
  InterpreterBuiltin
  MLIRArithDialect
  MLIRAsmParser
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIndexDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRUBDialect
  LLVMSupport)
