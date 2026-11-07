add_library(InterpreterSCF SCFEval.cpp)

target_link_libraries(InterpreterSCF PUBLIC
  InterpreterInterfaces
  InterpreterBuiltin
  MLIRSCFDialect)
