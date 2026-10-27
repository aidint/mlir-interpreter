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
  /// No lookup completed: the operation isn't cacheable, or its eligibility
  /// check or hashing signaled.
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
  // TODO: Remove `CacheOutcome` in favor of a better way to report lookups.
  Answer query(Value value, CacheOutcome &outcome);
  FailureOr<func::FuncOp> specialize(func::CallOp call,
                                     ArrayRef<std::optional<EvalValue>> args);

private:
  friend class BlockBinding;
  friend class EvalScope;

  /// This class represents the arguments of one block, bound for one loop
  /// iteration.
  struct BindingFrame {
    Block *block;
    /// Values defined in `block`'s region that were memoized in this
    /// iteration.
    SmallVector<Value> memoized;
  };

  EvalValueStorageAllocator &getQueryAllocator() { return *queryAllocator; }
  Answer queryNested(Value value);
  /// Binds `block`'s arguments to `args` for one iteration, or returns false
  /// when the budget ran out.
  bool bind(Block *block, ArrayRef<Answer> args);
  /// Unbinds the innermost frame, which must belong to `block`, and drops the
  /// answers memoized in it.
  void unbind(Block *block);
  /// Returns the answer for `value`, reusing one completed earlier in this
  /// query.
  Answer evaluate(Value value);
  /// Memoizes `answer` until the innermost frame whose region contains
  /// `value`'s definition ends, or for the whole query outside every frame.
  void memoize(Value value, const Answer &answer);
  /// Computes the answer for `value`, setting `outcome`, if given, to the
  /// lookup outcome of its operation.
  // TODO: Remove `CacheOutcome` in favor of a better way to report lookups.
  Answer compute(Value value, CacheOutcome *outcome = nullptr);
  /// Converts the attribute a constant-like op folds to, or returns null when
  /// it has no value. Constants have no operands, so they cost no step and
  /// never signal, and only cacheable attributes are looked up in the cache.
  EvalValue evaluateConstant(Operation *op, CacheOutcome *outcome);

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
  /// The blocks bound in the current query, most recent last.
  SmallVector<BindingFrame> frames;
  DenseMap<BlockArgument, Answer> bindings;
};

StringRef stringifyEvalStatus(EvalStatus status);

} // namespace mlir::interpreter

#endif
