add_library(InterpreterInterfaces
  EvalValueStorageAllocator.cpp
  EvalContext.cpp
  Interfaces/EquatableEvalValueInterface.cpp
  Interfaces/EvaluableOpInterface.cpp
  Interfaces/EvaluableAttrInterface.cpp)

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

add_subdirectory(Dialects)
