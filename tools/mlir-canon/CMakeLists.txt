add_executable(mlir-canon mlir-canon.cpp)

target_link_libraries(mlir-canon PRIVATE
  mlir-submodule
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
  LLVMSupport)
