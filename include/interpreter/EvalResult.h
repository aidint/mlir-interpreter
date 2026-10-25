#ifndef INTERPRETER_EVALRESULT_H
#define INTERPRETER_EVALRESULT_H

#include <cassert>
#include <type_traits>
#include <utility>

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

/// This class represents the outcome of an evaluation, a hash or a cache-key
/// comparison: a status and, once completed, a value.
template <typename T> struct EvalResult {
  EvalStatus status;
  /// Meaningful only when completed. A signal holds a default-constructed `T`,
  /// so `T` must be default-constructible.
  T value{};

  /// Builds a signal, so a cache-key method can return an operand's signal
  /// with `return answer.status;`.
  EvalResult(EvalStatus signal) : status(signal) {
    assert(signal != EvalStatus::Completed && "only signals omit the value");
  }
  /// Builds a completed result, so a method can `return value;`.
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U &&, T>>>
  EvalResult(U &&value)
      : status(EvalStatus::Completed), value(std::forward<U>(value)) {}

  bool completed() const { return status == EvalStatus::Completed; }

  const T &getValue() const {
    assert(completed() && "only completed results have a value");
    return value;
  }
};

}

#endif
