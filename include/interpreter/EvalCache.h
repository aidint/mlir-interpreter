#ifndef INTERPRETER_EVALCACHE_H
#define INTERPRETER_EVALCACHE_H

#include "interpreter/EvalContext.h"
#include "interpreter/EvalResult.h"
#include "interpreter/EvalValueStorageAllocator.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"

#include <optional>
#include <type_traits>

namespace mlir {
class Operation;
}

namespace mlir::interpreter {

class Engine;
class EvaluableOpInterface;

struct CacheEntry {
  Operation *op;
  MutableArrayRef<std::optional<Answer>> results;
};

static_assert(std::is_trivially_destructible_v<CacheEntry>,
              "cache entries are bump allocated and never destroyed");

struct CacheLookup {
  llvm::hash_code key;
  CacheEntry *entry;
};

class EvalCache {
public:
  explicit EvalCache(EvalContext &ctx) : allocator(ctx, true) {}

  void beginQuery() { allocator.beginRetaining(); }
  void clear() {
    entries.clear();
    entryAllocator.Reset();
    allocator.beginRetaining();
  }

  /// Finds the entry for `op` or for an operation equivalent to it. `op` must
  /// implement `EvaluableOpInterface`.
  EvalResult<CacheLookup> lookup(Operation *op, Engine &engine);
  /// Fills the entry for `op` or for an equivalent operation, creating one if
  /// none exists. `op` must implement `EvaluableOpInterface`.
  EvalResult<CacheEntry *> insert(Operation *op, ArrayRef<Answer> answers,
                                  Engine &engine);
  EvalResult<CacheEntry *> insert(Operation *op, llvm::hash_code key,
                                  ArrayRef<Answer> answers, Engine &engine);
  void fill(CacheEntry *entry, ArrayRef<Answer> answers);

private:
  EvalResult<CacheEntry *> findEquivalent(EvaluableOpInterface op,
                                          llvm::hash_code key, Engine &engine);

  EvalValueStorageAllocator allocator;
  DenseMap<llvm::hash_code, SmallVector<CacheEntry *>> entries;
  llvm::BumpPtrAllocator entryAllocator;
};

} // namespace mlir::interpreter

#endif
