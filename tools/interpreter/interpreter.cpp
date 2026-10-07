#include "interpreter/Dialects/Arith/ArithEval.h"
#include "interpreter/Dialects/Builtin/BuiltinAttrEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Engine.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Unicode.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace cl = llvm::cl;

using namespace mlir;
using namespace mlir::interpreter;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<uint64_t>
    budget("budget", cl::desc("Evaluation step budget of each reported value"),
           cl::init(1000));

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    cl::desc("Allow operations from unregistered dialects"), cl::init(false));

/// Returns `@main` if it is a straight-line function without arguments or
/// calls, the only entry points the runner supports so far.
static FailureOr<func::FuncOp> getEntryPoint(ModuleOp module) {
  auto main = module.lookupSymbol<func::FuncOp>("main");
  if (!main)
    return emitError(module.getLoc(), "no 'func.func @main' entry point");
  if (main.isExternal())
    return main.emitError("entry point has no body");
  if (main.getNumArguments() != 0)
    return main.emitError("entry point must not take arguments");
  if (!main.getBody().hasOneBlock())
    return main.emitError("control flow is not supported");
  for (Operation &op : main.getBody().front()) {
    if (isa<CallOpInterface>(op))
      return op.emitError("calls are not supported");
    if (op.getNumRegions() != 0 || op.getNumSuccessors() != 0)
      return op.emitError("control flow is not supported");
  }
  return main;
}

/// Prints `text` left-aligned in a column of `width` display columns, keeping
/// at least one space before the next column.
static void printColumn(raw_ostream &os, StringRef text, int width) {
  os << text;
  os.indent(std::max(width - llvm::sys::unicode::columnWidthUTF8(text), 1));
}

/// Returns the name `result` has in the source, e.g. `%a`, or `%pair#1` for a
/// result in a group.
static std::string getSourceName(OpResult result,
                                 const AsmParserState &parserState) {
  const auto *def = parserState.getOpDef(result.getOwner());
  assert(def && "every operation is parsed from the source");
  unsigned number = result.getResultNumber();
  const auto &groups = def->resultGroups;
  auto group = llvm::find_if(llvm::reverse(groups), [&](const auto &group) {
    return group.startIndex <= number;
  });
  assert(group != groups.rend() && "every result belongs to a group");
  llvm::SMRange range = group->definition.loc;
  std::string name(range.Start.getPointer(),
                   range.End.getPointer() - range.Start.getPointer());
  unsigned end = group == groups.rbegin() ? result.getOwner()->getNumResults()
                                          : std::prev(group)->startIndex;
  if (end - group->startIndex > 1)
    name += "#" + std::to_string(number - group->startIndex);
  return name;
}

static std::string printEvaluated(const Answer &answer) {
  if (answer.status != EvalStatus::Completed)
    return "—";
  const auto &value = answer.getValue();
  if (!value)
    return "unknown";
  if (auto integer = dyn_cast<IntEvalValue>(*value))
    return llvm::toString(integer.getValue(), 10, /*Signed=*/true);
  return "known";
}

static StringRef printCacheOutcome(CacheOutcome outcome) {
  switch (outcome) {
  case CacheOutcome::None:
    return "—";
  case CacheOutcome::Hit:
    return "hit";
  case CacheOutcome::Miss:
    return "miss";
  }
  llvm_unreachable("unhandled CacheOutcome");
}

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(
      argc, argv,
      "Query every result in @main in order, reporting cache outcomes\n");

  DialectRegistry registry;
  registry
      .insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
              index::IndexDialect, scf::SCFDialect, ub::UBDialect>();
  registerArithEvalExternalModels(registry);
  registerBuiltinAttrEvalExternalModels(registry);

  MLIRContext context(registry);
  context.allowUnregisteredDialects(allowUnregisteredDialects);
  context.printOpOnDiagnostic(false);

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  // Parsing records the source ranges of SSA names, so rows can show them.
  Block block;
  AsmParserState parserState;
  if (failed(parseAsmSourceFile(sourceMgr, &block, ParserConfig(&context),
                                &parserState)))
    return 1;
  OwningOpRef<ModuleOp> module =
      mlir::detail::constructContainerOpForParserIfNecessary<ModuleOp>(
          &block, &context,
          FileLineColLoc::get(&context, inputFilename, /*line=*/1,
                              /*column=*/1));
  if (!module)
    return 1;
  FailureOr<func::FuncOp> main = getEntryPoint(*module);
  if (failed(main))
    return 1;

  EvalContext evalContext(&context);
  registerBuiltinEvalValues(evalContext);
  // One engine shares the context's cache across every reported value, and
  // each query gets a fresh budget.
  Engine engine(evalContext, budget);

  constexpr int valueWidth = 8, evaluatedWidth = 12, cacheWidth = 9;
  raw_ostream &os = llvm::outs();
  printColumn(os, "value", valueWidth);
  printColumn(os, "evaluated", evaluatedWidth);
  printColumn(os, "cache", cacheWidth);
  os << "status\n";

  for (Operation &op : main->getBody().front()) {
    for (OpResult result : op.getResults()) {
      CacheOutcome outcome;
      Answer answer = engine.query(result, outcome);
      printColumn(os, getSourceName(result, parserState), valueWidth);
      printColumn(os, printEvaluated(answer), evaluatedWidth);
      printColumn(os, printCacheOutcome(outcome), cacheWidth);
      os << stringifyEvalStatus(answer.status) << "\n";
    }
  }
  return 0;
}
