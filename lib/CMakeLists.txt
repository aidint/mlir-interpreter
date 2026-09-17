add_library(MLIRInterpreterInterfaces EvaluableOpInterface.cpp)

add_dependencies(MLIRInterpreterInterfaces MLIREvaluableOpInterfaceIncGen)

target_include_directories(MLIRInterpreterInterfaces PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}/include")

target_link_libraries(MLIRInterpreterInterfaces PUBLIC
  mlir-submodule
  MLIRIR
  MLIRSupport)

add_library(MLIRInterpreter Engine.cpp)

target_link_libraries(MLIRInterpreter PUBLIC
  MLIRInterpreterInterfaces
  MLIRFuncDialect)

add_library(MLIRInterpreterArith ArithEval.cpp)

target_link_libraries(MLIRInterpreterArith PUBLIC
  MLIRInterpreterInterfaces
  MLIRArithDialect)
