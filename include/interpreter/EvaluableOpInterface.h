#ifndef INTERPRETER_EVALUABLEOPINTERFACE_H
#define INTERPRETER_EVALUABLEOPINTERFACE_H

#include "interpreter/EvalValue.h"
#include "interpreter/EvalValueStorageAllocator.h"

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

class Engine;

/// The view an operation's evaluation gets of the engine that drives it.
class EvalScope {
public:
  explicit EvalScope(Engine &engine) : engine(engine) {}

  Answer query(Value value);
  SmallVector<Answer> walk(Region &region, ArrayRef<Answer> args);
  bool isOrdered() const;
  EvalSession &getSession();

private:
  Engine &engine;
};

} // namespace mlir::interpreter

#include "interpreter/EvaluableOpInterface.h.inc"

#endif
