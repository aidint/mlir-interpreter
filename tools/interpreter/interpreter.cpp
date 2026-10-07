#include "interpreter/Tools/Runner.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
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

static cl::opt<std::string> queryName(
    "query",
    cl::desc("Also query the value with this source name, e.g. %b or %arg0, "
             "through the same engine, and print it in its own table"),
    cl::value_desc("name"));

static cl::opt<bool>
    queryBefore("query-before",
                cl::desc("Make the --query query before the rows instead of "
                         "after them"),
                cl::init(false));

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

/// Prints `text` left-aligned in a column of `width` display columns, keeping
/// at least one space before the next column.
static void printColumn(raw_ostream &os, StringRef text, int width) {
  os << text;
  os.indent(std::max(width - llvm::sys::unicode::columnWidthUTF8(text), 1));
}

/// Prints `rows` as a table whose first column is headed `nameHeader`.
static void printTable(raw_ostream &os, StringRef nameHeader,
                       ArrayRef<ReportRow> rows) {
  constexpr int valueWidth = 8, evaluatedWidth = 12, cacheWidth = 9;
  printColumn(os, nameHeader, valueWidth);
  printColumn(os, "evaluated", evaluatedWidth);
  printColumn(os, "cache", cacheWidth);
  os << "status\n";
  for (const ReportRow &row : rows) {
    printColumn(os, row.name, valueWidth);
    printColumn(os, row.evaluated, evaluatedWidth);
    printColumn(os, row.cache, cacheWidth);
    os << row.status << "\n";
  }
}

/// Runs `@main` of the module in `sourceMgr` and prints its report to `os`.
/// A named query gets its own table, printed in the order it was made.
static LogicalResult run(llvm::SourceMgr &sourceMgr, MLIRContext &context,
                         raw_ostream &os) {
  std::optional<NamedQuery> query;
  if (!queryName.empty())
    query = NamedQuery{queryName, queryBefore ? QueryPosition::Before
                                              : QueryPosition::After};
  FailureOr<RunReport> report = runMain(sourceMgr, context, budget, query);
  if (failed(report))
    return failure();

  if (query && query->position == QueryPosition::Before) {
    printTable(os, "query", *report->query);
    os << "\n";
  }
  printTable(os, "value", report->rows);
  if (query && query->position == QueryPosition::After) {
    os << "\n";
    printTable(os, "query", *report->query);
  }
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
  registerRunnerDialects(registry);

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
