#ifndef INTERPRETER_ENGINE_H
#define INTERPRETER_ENGINE_H

#include "interpreter/EvalCache.h"
#include "interpreter/EvalContext.h"
#include "interpreter/EvalValueStorageAllocator.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstdint>
#include <optional>

namespace mlir::interpreter {

/// The outcome of the cache lookup for a queried value's own operation.
enum class CacheOutcome {
  /// No lookup completed: the operation isn't cached or hashing signaled.
  None,
  /// The lookup found a completed answer for the value.
  Hit,
  /// The lookup completed without one, so the value was evaluated.
  Miss,
};

class Engine {
public:
  Engine(EvalContext &ctx, uint64_t budget)
      : ctx(ctx), budget(budget), answerAllocator(ctx, false) {}

  EvalContext &getContext() { return ctx; }

  Answer query(Value value);
  /// Queries `value` and sets `outcome` to the lookup outcome of its own
  /// operation; lookups made by nested queries don't affect it.
  Answer query(Value value, CacheOutcome &outcome);
  FailureOr<func::FuncOp> specialize(func::CallOp call,
                                     ArrayRef<std::optional<EvalValue>> args);

private:
  friend class EvalScope;

  EvalValueStorageAllocator &getQueryAllocator() { return *queryAllocator; }
  Answer queryNested(Value value);
  /// Returns the answer for `value`, reusing one completed earlier in this
  /// query.
  Answer evaluate(Value value);
  /// Computes the answer for `value`, setting `outcome`, if given, to the
  /// lookup outcome of its operation.
  Answer compute(Value value, CacheOutcome *outcome = nullptr);

  EvalContext &ctx;
  uint64_t budget;
  uint64_t remaining = 0;
  EvalValueStorageAllocator answerAllocator;
  std::optional<EvalValueStorageAllocator> queryAllocator;
  /// Operations being evaluated in the current query.
  llvm::SmallPtrSet<Operation *, 8> active;
  /// Completed answers of the current query; cleared with the query allocator
  /// their values may live in.
  DenseMap<Value, Answer> queryAnswers;
};

StringRef stringifyEvalStatus(EvalStatus status);

} // namespace mlir::interpreter

#endif
