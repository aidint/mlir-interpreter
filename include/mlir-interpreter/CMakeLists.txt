include_directories("${LLVM_PROJECT_SOURCE_DIR}/mlir/include")

add_mlir_interface(EvaluableOpInterface)

set(LLVM_TARGET_DEFINITIONS EvaluableAttrInterface.td)
mlir_tablegen(EvaluableAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(EvaluableAttrInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIREvaluableAttrInterfaceIncGen)
