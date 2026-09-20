add_library(InterpreterInterfaces
  EvalContext.cpp
  EvalValue.cpp
  EvaluableOpInterface.cpp
  EvaluableAttrInterface.cpp)

add_dependencies(InterpreterInterfaces
  MLIREvaluableOpInterfaceIncGen
  MLIREvaluableAttrInterfaceIncGen)

target_include_directories(InterpreterInterfaces PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}/include")

target_link_libraries(InterpreterInterfaces PUBLIC
  mlir-submodule
  MLIRIR
  MLIRSupport)

add_library(Interpreter Engine.cpp)

target_link_libraries(Interpreter PUBLIC
  InterpreterInterfaces
  MLIRFuncDialect)

add_library(InterpreterBuiltin BuiltinAttrEval.cpp BuiltinEvalValues.cpp)

target_link_libraries(InterpreterBuiltin PUBLIC
  InterpreterInterfaces)

add_library(InterpreterArith ArithEval.cpp)

target_link_libraries(InterpreterArith PUBLIC
  InterpreterInterfaces
  InterpreterBuiltin
  MLIRArithDialect)
