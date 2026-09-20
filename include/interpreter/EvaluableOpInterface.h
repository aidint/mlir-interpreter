#ifndef INTERPRETER_EVALUABLEOPINTERFACE_H
#define INTERPRETER_EVALUABLEOPINTERFACE_H

#include "interpreter/EvalValue.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::interpreter {

enum class AnswerKind { Known, Unknown, Exhausted, NeedsOrder };

struct Answer {
  AnswerKind kind;
  std::optional<EvalValue> value;
};

class EvalScope {
public:
  virtual ~EvalScope() = default;

  virtual Answer query(Value value) = 0;
  virtual SmallVector<Answer> walk(Region &region, ArrayRef<Answer> args) = 0;
  virtual bool isOrdered() const = 0;
  virtual EvalSession &getSession() = 0;
};

} // namespace mlir::interpreter

#include "interpreter/EvaluableOpInterface.h.inc"

#endif
