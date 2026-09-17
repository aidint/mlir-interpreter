#ifndef MLIR_INTERPRETER_EVALUABLEOPINTERFACE_H
#define MLIR_INTERPRETER_EVALUABLEOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <any>
#include <optional>

namespace mlir::interpreter {

struct InterpreterValue {
  std::any payload;
};

enum class AnswerKind { Known, Unknown, Exhausted, NeedsOrder };

struct Answer {
  AnswerKind kind;
  std::optional<InterpreterValue> value;
};

class EvalContext {
public:
  virtual ~EvalContext() = default;

  virtual Answer query(Value value) = 0;
  virtual SmallVector<Answer> walk(Region &region, ArrayRef<Answer> args) = 0;
  virtual bool isOrdered() const = 0;
};

} // namespace mlir::interpreter

#include "mlir-interpreter/EvaluableOpInterface.h.inc"

#endif
