#ifndef INTERPRETER_EVALRESULT_H
#define INTERPRETER_EVALRESULT_H

#include "interpreter/EvalValue.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <optional>

namespace mlir::interpreter {

/// The outcome of evaluating a value, hashing or comparing a cache key. Only
/// `Completed` describes the program; the others describe one run and are never
/// cached.
enum class EvalStatus {
  /// The evaluation finished. Its value may still be unknown.
  Completed,
  /// The query's budget ran out.
  Exhausted,
  /// The answer depends on execution order, which the current walk doesn't fix.
  NeedsOrder,
  /// Evaluating this value re-entered an operation whose evaluation is still in
  /// progress in the current query.
  Cycle,
};

template <typename T> struct EvalResult;
using Answer = EvalResult<EvalValue>;

/// This class represents the outcome of an evaluation, a hash or a cache-key
/// comparison: a status and, once completed, a value that may be unknown.
template <typename T> struct EvalResult {
  EvalStatus status;
  std::optional<T> value;

  /// Builds a signal, so a cache-key method can return an operand's signal
  /// with `return answer.status;`.
  EvalResult(EvalStatus signal) : status(signal) {
    assert(signal != EvalStatus::Completed && "only signals omit the value");
  }
  EvalResult(EvalStatus status, std::optional<T> value)
      : status(status), value(std::move(value)) {}

  bool completed() const { return status == EvalStatus::Completed; }

  const std::optional<T> &getValue() const {
    assert((completed() || !value) &&
           "incomplete evaluations cannot have a value");
    return value;
  }

  /// Returns this signal for every result of `op`, so an evaluation function
  /// can propagate an operand's `Exhausted`, `NeedsOrder` or `Cycle`.
  auto propagate(Operation *op) const {
    // The return type, `SmallVector<Answer>`, is deduced: its default inline
    // size needs the complete `Answer`, which this class is still defining.
    assert(!completed() && "only signals propagate");
    return SmallVector<Answer>(op->getNumResults(), Answer(status));
  }
};

}

#endif
