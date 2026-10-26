#ifndef INTERPRETER_ANSWER_H
#define INTERPRETER_ANSWER_H

#include "interpreter/EvalResult.h"
#include "interpreter/EvalValue.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>

namespace mlir::interpreter {

/// This class represents the answer for a value: a signal, or a completed
/// value that is unknown when null.
struct Answer : EvalResult<EvalValue> {
  using EvalResult::EvalResult;

  static Answer unknown() { return EvalValue(); }

  /// Returns whether this answer completed without a value. A signal is not
  /// unknown.
  bool isUnknown() const { return completed() && !value; }

  /// Returns this signal for every result of `op`, so an evaluation function
  /// can propagate an operand's `Exhausted`, `NeedsOrder` or `Cycle`.
  auto propagate(Operation *op) const {
    // The return type, `SmallVector<Answer>`, is deduced: its default inline
    // size needs the complete `Answer`, which only the body can see.
    assert(!completed() && "only signals propagate");
    return SmallVector<Answer>(op->getNumResults(), Answer(status));
  }
};

} // namespace mlir::interpreter

#endif
