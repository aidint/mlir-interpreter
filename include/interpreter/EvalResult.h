#ifndef INTERPRETER_EVALRESULT_H
#define INTERPRETER_EVALRESULT_H

#include "interpreter/EvalValue.h"

#include <cassert>
#include <optional>

namespace mlir::interpreter {

enum class EvalStatus { Completed, Exhausted, NeedsOrder };

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
