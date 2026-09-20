#ifndef INTERPRETER_ENGINE_H
#define INTERPRETER_ENGINE_H

#include "interpreter/EvalCache.h"
#include "interpreter/EvalContext.h"
#include "interpreter/EvaluableOpInterface.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::interpreter {

class EngineScope;

class Engine {
public:
  Engine(EvalContext &ctx, uint64_t budget);

  void registerType(TypeID type);

  EvalContext &getContext() { return ctx; }

  Answer query(Value value, EvalBasket &basket);
  FailureOr<func::FuncOp> specialize(func::CallOp call,
                                     ArrayRef<std::optional<EvalValue>> args);

private:
  friend class EngineScope;

  EvalSession &getSession() { return *session; }
  Answer queryNested(Value value);
  Answer evaluate(Value value);

  EvalContext &ctx;
  uint64_t budget;
  uint64_t remaining = 0;
  std::optional<EvalSession> session;
};

StringRef stringifyAnswerKind(AnswerKind kind);

} // namespace mlir::interpreter

#endif
