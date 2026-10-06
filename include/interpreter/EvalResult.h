#ifndef INTERPRETER_EVALRESULT_H
#define INTERPRETER_EVALRESULT_H

#include "interpreter/EvalValue.h"

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

template <typename T> struct EvalResult {
  EvalStatus status;
  std::optional<T> value;

  const std::optional<T> &getValue() const {
    assert((status == EvalStatus::Completed || !value) &&
           "incomplete evaluations cannot have a value");
    return value;
  }
};

using Answer = EvalResult<EvalValue>;

}

#endif
