#ifndef INTERPRETER_ENGINE_H
#define INTERPRETER_ENGINE_H

#include "interpreter/EvalCache.h"
#include "interpreter/EvalContext.h"
#include "interpreter/EvalValueStorageAllocator.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <cstdint>
#include <optional>

namespace mlir::interpreter {

class Engine {
public:
  Engine(EvalContext &ctx, EvalCache &cache, uint64_t budget)
      : ctx(ctx), cache(cache), budget(budget), answerAllocator(ctx, false) {}

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
  EvalCache &cache;
  uint64_t budget;
  uint64_t remaining = 0;
  EvalValueStorageAllocator answerAllocator;
  std::optional<EvalValueStorageAllocator> queryAllocator;
  /// Operations being evaluated in the current query.
  llvm::SmallPtrSet<Operation *, 8> active;
};

StringRef stringifyEvalStatus(EvalStatus status);

} // namespace mlir::interpreter

#endif
