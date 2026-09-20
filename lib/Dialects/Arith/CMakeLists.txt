add_library(InterpreterArith ArithEval.cpp)

target_link_libraries(InterpreterArith PUBLIC
  InterpreterInterfaces
  InterpreterBuiltin
  MLIRArithDialect)
