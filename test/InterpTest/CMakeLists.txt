# The `interp_test` dialect: operations the tests need that the linked dialects
# lack.
include_directories("${LLVM_PROJECT_SOURCE_DIR}/mlir/include")

set(LLVM_TARGET_DEFINITIONS InterpTestOps.td)
mlir_tablegen(InterpTestOps.h.inc -gen-op-decls)
mlir_tablegen(InterpTestOps.cpp.inc -gen-op-defs)
mlir_tablegen(InterpTestDialect.h.inc -gen-dialect-decls)
mlir_tablegen(InterpTestDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(InterpTestIncGen)

add_library(InterpTest InterpTestDialect.cpp)

add_dependencies(InterpTest InterpTestIncGen)

target_include_directories(InterpTest PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(InterpTest PUBLIC
  mlir-submodule
  MLIRBytecodeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces)
