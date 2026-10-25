#ifndef INTERPRETER_EVALRESULT_H
#define INTERPRETER_EVALRESULT_H

#include "interpreter/EvalValue.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <optional>
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

struct Answer;

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

  /// Returns this signal for every result of `op`, so an evaluation function
  /// can propagate an operand's `Exhausted`, `NeedsOrder` or `Cycle`.
  auto propagate(Operation *op) const;
};

/// This class represents the answer for a value: a signal, or a completed
/// value that is unknown when absent.
struct Answer : EvalResult<std::optional<EvalValue>> {
  using EvalResult::EvalResult;

  static Answer unknown() { return std::nullopt; }

  /// Returns whether this answer completed without a value. A signal is not
  /// unknown.
  bool isUnknown() const { return completed() && !value; }
};

// The return type, `SmallVector<Answer>`, is deduced and the body follows
// `Answer`: its default inline size needs the complete `Answer`, which derives
// from `EvalResult`.
template <typename T> auto EvalResult<T>::propagate(Operation *op) const {
  assert(!completed() && "only signals propagate");
  return SmallVector<Answer>(op->getNumResults(), Answer(status));
}

}

#endif
