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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"

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

static cl::opt<bool> splitInputFile(
    "split-input-file",
    cl::desc("Run each chunk of the input, split at '// -----', on its own"),
    cl::init(false));

static cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    cl::desc("Check that emitted diagnostics match expected-* comments"),
    cl::init(false));

/// Returns `@main` if it has a body to run.
static FailureOr<func::FuncOp> getEntryPoint(ModuleOp module) {
  auto main = module.lookupSymbol<func::FuncOp>("main");
  if (!main)
    return emitError(module.getLoc(), "no 'func.func @main' entry point");
  if (main.isExternal())
    return main.emitError("entry point has no body");
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
  if (auto integer = dyn_cast<IntEvalValue>(*value)) {
    const APInt &bits = integer.getValue();
    // Printed as signed, an i1 `true` would read as -1.
    if (bits.getBitWidth() == 1)
      return bits.isOne() ? "true" : "false";
    return llvm::toString(bits, 10, /*Signed=*/true);
  }
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

/// Runs `@main` of the module in `sourceMgr` and prints its report to `os`.
static LogicalResult run(llvm::SourceMgr &sourceMgr, MLIRContext &context,
                         raw_ostream &os) {
  // Parsing records the source ranges of SSA names, so rows can show them.
  Block block;
  AsmParserState parserState;
  if (failed(parseAsmSourceFile(sourceMgr, &block, ParserConfig(&context),
                                &parserState)))
    return failure();
  StringRef bufferName = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())
                             ->getBufferIdentifier();
  OwningOpRef<ModuleOp> module =
      mlir::detail::constructContainerOpForParserIfNecessary<ModuleOp>(
          &block, &context,
          FileLineColLoc::get(&context, bufferName, /*line=*/1, /*column=*/1));
  if (!module)
    return failure();
  FailureOr<func::FuncOp> main = getEntryPoint(*module);
  if (failed(main))
    return failure();

  EvalContext evalContext(&context);
  registerBuiltinEvalValues(evalContext);
  // One engine shares the context's cache across every reported value, and
  // each query gets a fresh budget.
  Engine engine(evalContext, budget);

  constexpr int valueWidth = 8, evaluatedWidth = 12, cacheWidth = 9;
  printColumn(os, "value", valueWidth);
  printColumn(os, "evaluated", evaluatedWidth);
  printColumn(os, "cache", cacheWidth);
  os << "status\n";

  // Every block and nested region is reported in source order. Values are
  // queried on demand, so a row doesn't depend on which branch would run.
  main->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (OpResult result : op->getResults()) {
      CacheOutcome outcome;
      Answer answer = engine.query(result, outcome);
      printColumn(os, getSourceName(result, parserState), valueWidth);
      printColumn(os, printEvaluated(answer), evaluatedWidth);
      printColumn(os, printCacheOutcome(outcome), cacheWidth);
      os << stringifyEvalStatus(answer.status) << "\n";
    }
  });
  return success();
}

/// Runs one chunk of the input in its own context. Under
/// `--verify-diagnostics`, succeeds when the diagnostics match the chunk's
/// `expected-*` comments instead.
static LogicalResult runChunk(std::unique_ptr<llvm::MemoryBuffer> chunk,
                              DialectRegistry &registry, raw_ostream &os) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(chunk), llvm::SMLoc());
  MLIRContext context(registry);
  context.allowUnregisteredDialects(allowUnregisteredDialects);
  context.printOpOnDiagnostic(false);
  if (verifyDiagnostics) {
    SourceMgrDiagnosticVerifierHandler handler(sourceMgr, &context);
    (void)run(sourceMgr, context, os);
    return handler.verify();
  }
  SourceMgrDiagnosticHandler handler(sourceMgr, &context);
  return run(sourceMgr, context, os);
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

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto processChunk = [&](std::unique_ptr<llvm::MemoryBuffer> chunk,
                          raw_ostream &os) {
    return runChunk(std::move(chunk), registry, os);
  };
  return failed(
             splitAndProcessBuffer(std::move(input), processChunk, llvm::outs(),
                                   splitInputFile ? kDefaultSplitMarker : ""))
             ? 1
             : 0;
}
