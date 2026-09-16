//===- mlir-canon.cpp - Parse an MLIR file and canonicalize it ------------===//
//
// Reads an MLIR file, runs the canonicalize pass over it, and prints the
// result.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <string>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    cl::desc("Allow operations from unregistered dialects"), cl::init(false));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv,
                              "Parse an MLIR file and canonicalize it\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::ub::UBDialect>();

  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects(allowUnregisteredDialects);

  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module)
    return 1;

  mlir::PassManager pm(&context);
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 1;
  pm.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(*module)))
    return 1;

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  module->print(output->os());
  output->keep();
  return 0;
}
