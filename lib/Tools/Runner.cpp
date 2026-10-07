#include "interpreter/Tools/Runner.h"

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

#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::interpreter;

void mlir::interpreter::registerRunnerDialects(DialectRegistry &registry) {
  registry
      .insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
              index::IndexDialect, scf::SCFDialect, ub::UBDialect>();
  registerArithEvalExternalModels(registry);
  registerBuiltinAttrEvalExternalModels(registry);
}

/// Returns `@main` if it has a body to run.
static FailureOr<func::FuncOp> getEntryPoint(ModuleOp module) {
  auto main = module.lookupSymbol<func::FuncOp>("main");
  if (!main)
    return emitError(module.getLoc(), "no 'func.func @main' entry point");
  if (main.isExternal())
    return main.emitError("entry point has no body");
  return main;
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

/// Returns the name `arg` has in the source, e.g. `%x`, or an empty string
/// when its block was built without naming it.
static std::string getSourceName(BlockArgument arg,
                                 const AsmParserState &parserState) {
  const auto *def = parserState.getBlockDef(arg.getOwner());
  if (!def || arg.getArgNumber() >= def->arguments.size())
    return {};
  llvm::SMRange range = def->arguments[arg.getArgNumber()].loc;
  if (!range.isValid())
    return {};
  return std::string(range.Start.getPointer(),
                     range.End.getPointer() - range.Start.getPointer());
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

/// Returns the value of `main` named `name` in the source, an operation result
/// or a block argument, or null.
static Value lookupValue(func::FuncOp main, StringRef name,
                         const AsmParserState &parserState) {
  Value found;
  main->walk([&](Block *block) {
    for (BlockArgument arg : block->getArguments()) {
      if (getSourceName(arg, parserState) == name) {
        found = arg;
        return WalkResult::interrupt();
      }
    }
    for (Operation &op : *block) {
      for (OpResult result : op.getResults()) {
        if (getSourceName(result, parserState) == name) {
          found = result;
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return found;
}

FailureOr<RunReport>
mlir::interpreter::runMain(llvm::SourceMgr &sourceMgr, MLIRContext &context,
                           uint64_t budget,
                           const std::optional<NamedQuery> &query) {
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
  Value queried;
  if (query) {
    queried = lookupValue(*main, query->name, parserState);
    if (!queried)
      return main->emitError()
             << "no value named '" << query->name << "' in @main";
  }

  EvalContext evalContext(&context);
  registerBuiltinEvalValues(evalContext);
  // One engine shares the context's cache across every reported value, and
  // each query gets a fresh budget.
  Engine engine(evalContext, budget);
  auto queryRow = [&](Value value, std::string name) -> ReportRow {
    CacheOutcome outcome;
    Answer answer = engine.query(value, outcome);
    return {std::move(name), printEvaluated(answer),
            printCacheOutcome(outcome), stringifyEvalStatus(answer.status)};
  };

  RunReport report;
  if (query && query->position == QueryPosition::Before)
    report.query = queryRow(queried, query->name);
  // Every block and nested region is reported in source order. Values are
  // queried on demand, so a row doesn't depend on which branch would run.
  main->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (OpResult result : op->getResults())
      report.rows.push_back(
          queryRow(result, getSourceName(result, parserState)));
  });
  if (query && query->position == QueryPosition::After)
    report.query = queryRow(queried, query->name);
  return report;
}
