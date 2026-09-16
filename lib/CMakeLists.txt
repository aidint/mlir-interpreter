add_library(MLIRInterpreter Engine.cpp)

target_include_directories(MLIRInterpreter PUBLIC
  "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(MLIRInterpreter PUBLIC
  mlir-submodule
  MLIRFuncDialect
  MLIRIR
  MLIRSupport)
