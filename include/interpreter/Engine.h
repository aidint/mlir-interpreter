#ifndef INTERPRETER_ENGINE_H
#define INTERPRETER_ENGINE_H

#include "interpreter/EvalContext.h"
#include "interpreter/EvalValueStorageAllocator.h"
#include "interpreter/EvaluableOpInterface.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include <cstdint>
#include <optional>

namespace mlir::interpreter {

class Engine {
public:
  Engine(EvalContext &ctx, uint64_t budget);

  EvalContext &getContext() { return ctx; }

  Answer query(Value value);
  FailureOr<func::FuncOp> specialize(func::CallOp call,
                                     ArrayRef<std::optional<EvalValue>> args);

private:
  friend class EvalScope;

  EvalValueStorageAllocator &getQueryAllocator() { return *queryAllocator; }
  Answer queryNested(Value value);
  Answer evaluate(Value value);

  EvalContext &ctx;
  uint64_t budget;
  uint64_t remaining = 0;
  EvalValueStorageAllocator answerAllocator;
  std::optional<EvalValueStorageAllocator> queryAllocator;
};

StringRef stringifyAnswerKind(AnswerKind kind);

} // namespace mlir::interpreter

#endif
