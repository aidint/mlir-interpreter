#ifndef MLIR_INTERPRETER_ENGINE_H
#define MLIR_INTERPRETER_ENGINE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <any>
#include <cstdint>
#include <functional>
#include <optional>

namespace mlir::interpreter {

struct InterpValue {
  std::any payload;
};

enum class AnswerKind { Known, Unknown, Exhausted, NeedsOrder };

struct Answer {
  AnswerKind kind;
  std::optional<InterpValue> value;
};

class Engine;

class EvalContext {
public:
  explicit EvalContext(Engine &engine);

  Answer query(Value value);
  SmallVector<Answer> walk(Region &region, ArrayRef<Answer> args);
  bool isOrdered() const;

private:
  Engine &engine;
};

using EvalFn = std::function<SmallVector<Answer>(Operation *, ArrayRef<Answer>,
                                                 EvalContext &)>;

class Engine {
public:
  explicit Engine(uint64_t budget);

  void registerOp(StringRef opName, EvalFn fn);
  void registerType(TypeID type);

  Answer query(Value value);
  FailureOr<func::FuncOp> specialize(func::CallOp call,
                                     ArrayRef<std::optional<InterpValue>> args);

private:
  uint64_t budget;
};

StringRef stringifyAnswerKind(AnswerKind kind);

} // namespace mlir::interpreter

#endif
