#include "interpreter/Dialects/Arith/ArithEval.h"
#include "interpreter/Dialects/Builtin/BuiltinAttrEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Engine.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<uint64_t> budget("budget", cl::desc("Evaluation step budget"),
                                cl::init(1000));

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    cl::desc("Allow operations from unregistered dialects"), cl::init(false));

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Query every SSA value in a file\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::index::IndexDialect,
                  mlir::scf::SCFDialect, mlir::ub::UBDialect>();
  mlir::interpreter::registerArithEvalExternalModels(registry);
  mlir::interpreter::registerBuiltinAttrEvalExternalModels(registry);

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

  mlir::interpreter::EvalContext evalContext(&context);
  mlir::interpreter::registerBuiltinEvalValues(evalContext);
  mlir::interpreter::Engine engine(evalContext, budget);
  mlir::AsmState asmState(*module);

  module->walk([&](mlir::Operation *op) {
    for (mlir::Value result : op->getResults()) {
      mlir::interpreter::EvalBasket basket(evalContext);
      mlir::interpreter::Answer answer = engine.query(result, basket);
      result.printAsOperand(llvm::outs(), asmState);
      llvm::outs() << " -> ";
      auto value =
          answer.value
              ? llvm::dyn_cast<mlir::interpreter::IntEvalValue>(*answer.value)
              : mlir::interpreter::IntEvalValue();
      if (value)
        llvm::outs() << value.getValue();
      else
        llvm::outs() << mlir::interpreter::stringifyAnswerKind(answer.kind);
      llvm::outs() << "\n";
    }
  });
  return 0;
}
