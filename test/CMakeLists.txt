add_executable(engine-lifetime-test EngineLifetime.cpp)

target_link_libraries(engine-lifetime-test PRIVATE
  Interpreter
  InterpreterArith
  InterpreterBuiltin
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport)

add_test(NAME engine-lifetime COMMAND engine-lifetime-test)
