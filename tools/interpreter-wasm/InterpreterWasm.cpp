#include "interpreter/Tools/Runner.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <emscripten/emscripten.h>

#include <cstdlib>
#include <cstring>
#include <string>

using namespace mlir;
using namespace mlir::interpreter;

namespace {
/// This struct represents a diagnostic emitted during a run, located by line
/// and column when it has a file location.
struct DiagnosticEntry {
  StringRef severity;
  unsigned line = 0;
  unsigned column = 0;
  std::string message;
};
} // namespace

static StringRef stringifySeverity(DiagnosticSeverity severity) {
  switch (severity) {
  case DiagnosticSeverity::Note:
    return "note";
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  case DiagnosticSeverity::Remark:
    return "remark";
  }
  llvm_unreachable("unhandled DiagnosticSeverity");
}

static DiagnosticEntry getEntry(const Diagnostic &diag) {
  DiagnosticEntry entry{stringifySeverity(diag.getSeverity()), 0, 0,
                        diag.str()};
  if (auto loc = diag.getLocation()->findInstanceOf<FileLineColLoc>()) {
    entry.line = loc.getLine();
    entry.column = loc.getColumn();
  }
  return entry;
}

static void writeRow(llvm::json::OStream &json, const ReportRow &row) {
  json.attribute("name", row.name);
  json.attribute("evaluated", row.evaluated);
  json.attribute("cache", row.cache);
  json.attribute("status", row.status);
}

/// Runs `@main` in `source`, plus `query` if given, with a fresh context and
/// cache, and writes the report to `os` as JSON:
///
///   {"rows": [{"name": "%a", "evaluated": "5", "cache": "miss",
///              "status": "completed"}],
///    "query": {"name": "%b", ..., "position": "after"},
///    "diagnostics": [{"severity": "error", "line": 3, "column": 5,
///                     "message": "..."}]}
///
/// Rows hold the tool's spelling of each column, and `query` is null without a
/// query. A diagnostic without a file location has line and column 0.
static void run(StringRef source, uint64_t budget,
                const std::optional<NamedQuery> &query, raw_ostream &os) {
  DialectRegistry registry;
  registerRunnerDialects(registry);
  MLIRContext context(registry);
  context.printOpOnDiagnostic(false);

  SmallVector<DiagnosticEntry> diagnostics;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    diagnostics.push_back(getEntry(diag));
    for (const Diagnostic &note : diag.getNotes())
      diagnostics.push_back(getEntry(note));
  });

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(source, "input.mlir"), llvm::SMLoc());
  FailureOr<RunReport> report = runMain(sourceMgr, context, budget, query);

  llvm::json::OStream json(os);
  json.object([&] {
    json.attributeArray("rows", [&] {
      if (failed(report))
        return;
      for (const ReportRow &row : report->rows)
        json.object([&] { writeRow(json, row); });
    });
    json.attributeBegin("query");
    if (succeeded(report) && report->query) {
      json.object([&] {
        writeRow(json, *report->query);
        json.attribute("position", query->position == QueryPosition::Before
                                       ? "before"
                                       : "after");
      });
    } else {
      json.value(nullptr);
    }
    json.attributeEnd();
    json.attributeArray("diagnostics", [&] {
      for (const DiagnosticEntry &diag : diagnostics) {
        json.object([&] {
          json.attribute("severity", diag.severity);
          json.attribute("line", diag.line);
          json.attribute("column", diag.column);
          json.attribute("message", diag.message);
        });
      }
    });
  });
}

extern "C" {

/// Runs `@main` in the NUL-terminated MLIR `source` and returns the report as
/// NUL-terminated JSON, which the caller releases with
/// `interpreter_free_report`. A non-null `queryName` also queries that result,
/// before the rows if `queryBefore` is set and after them otherwise.
EMSCRIPTEN_KEEPALIVE char *interpreter_run(const char *source, uint64_t budget,
                                           const char *queryName,
                                           bool queryBefore) {
  std::optional<NamedQuery> query;
  if (queryName)
    query = NamedQuery{queryName, queryBefore ? QueryPosition::Before
                                              : QueryPosition::After};
  std::string report;
  llvm::raw_string_ostream os(report);
  run(source, budget, query, os);
  return strdup(report.c_str());
}

EMSCRIPTEN_KEEPALIVE void interpreter_free_report(char *report) {
  std::free(report);
}
}
