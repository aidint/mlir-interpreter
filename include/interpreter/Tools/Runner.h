#ifndef INTERPRETER_TOOLS_RUNNER_H
#define INTERPRETER_TOOLS_RUNNER_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
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

/// When a named query is made, relative to the rows of `@main`.
enum class QueryPosition { Before, After };

/// This struct represents a direct query of one value of `@main` by its source
/// name: an operation result or a block argument. It is made through the same
/// engine as the rows, so it shares their cache.
struct NamedQuery {
  /// The value's name in the source, e.g. `%b`, `%pair#1` for a result in a
  /// group, or `%arg0` for a block argument.
  std::string name;
  QueryPosition position = QueryPosition::After;
};

/// This struct represents the report of a run.
struct RunReport {
  /// One row per result of `@main`, in source order.
  SmallVector<ReportRow> rows;
  /// The row of the named query, when one was requested.
  std::optional<ReportRow> query;
};

/// Registers the dialects and evaluation models that runs support.
void registerRunnerDialects(DialectRegistry &registry);

/// Parses the main buffer of `sourceMgr` and queries every result of its
/// `@main`, across all blocks and nested regions in source order, plus `query`
/// before or after them if given. Each query gets `budget` steps, and all of
/// them share one cache. Emits a diagnostic and fails when parsing fails,
/// `@main` is missing or has no body, or no value has the queried name.
FailureOr<RunReport>
runMain(llvm::SourceMgr &sourceMgr, MLIRContext &context, uint64_t budget,
        const std::optional<NamedQuery> &query = std::nullopt);

} // namespace mlir::interpreter

#endif
