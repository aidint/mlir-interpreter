#ifndef INTERPRETER_TOOLS_RUNNER_H
#define INTERPRETER_TOOLS_RUNNER_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace mlir::interpreter {

/// This struct represents one row of a run's report: a result of `@main` and
/// how its query went, spelled as the `interpreter` tool prints it.
struct ReportRow {
  /// The result's name in the source, e.g. `%a`, or `%pair#1` in a group.
  std::string name;
  /// The value, `unknown`, or `—` when the query didn't complete.
  std::string evaluated;
  /// `hit` or `miss`, or `—` when no lookup completed.
  StringRef cache;
  /// The evaluation status, e.g. `completed` or `exhausted`.
  StringRef status;
};

/// Registers the dialects and evaluation models that runs support.
void registerRunnerDialects(DialectRegistry &registry);

/// Parses the main buffer of `sourceMgr` and queries every result of its
/// `@main`, across all blocks and nested regions in source order. Each query
/// gets `budget` steps, and all of them share one cache. Emits a diagnostic and
/// fails when parsing fails or `@main` is missing or has no body.
FailureOr<SmallVector<ReportRow>> runMain(llvm::SourceMgr &sourceMgr,
                                          MLIRContext &context,
                                          uint64_t budget);

} // namespace mlir::interpreter

#endif
