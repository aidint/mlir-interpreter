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

class EvalScope;
class EvaluableOpInterface;

/// This class represents the cached answers of an operation and of the
/// operations equivalent to it, with one slot per result. An empty slot has no
/// completed answer yet and can be filled later.
struct CacheEntry {
  Operation *op;
  MutableArrayRef<std::optional<Answer>> results;
};

static_assert(std::is_trivially_destructible_v<CacheEntry> &&
                  std::is_trivially_destructible_v<std::optional<Answer>>,
              "cache entries and their result slots are bump allocated and "
              "never destroyed");

/// This class represents the outcome of a cache lookup: the operation's key,
/// the matching entry (null on a miss), and the number of candidates in the
/// key's bucket that were already examined, so insertion can resume the search.
struct CacheLookup {
  llvm::hash_code key;
  CacheEntry *entry;
  size_t searched;
};

/// This class represents the context cache: the answers operations completed,
/// kept across queries. Each `EvalContext` owns one, shared by every `Engine`
/// on that context, so an answer one query computes is reused by later queries
/// and other engines, and cached values live as long as the context. Entries
/// are bucketed by `getHash` and matched with `isEqual`, so equivalent
/// operations share one entry. Only `Engine` fills it, inside a query, because
/// filling copies values out of the query allocator and dedups them by
/// address, which is only valid while that query runs.
class EvalCache {
public:
  void clear() {
    entries.clear();
    entryAllocator.Reset();
    allocator.beginRetaining();
  }

private:
  friend class EvalContext;
  friend class Engine;

  explicit EvalCache(EvalContext &ctx) : allocator(ctx, true) {}

  void beginQuery() { allocator.beginRetaining(); }

  /// Finds the entry for `op` or for an operation equivalent to it. `op` must
  /// implement `EvaluableOpInterface`.
  EvalResult<CacheLookup> lookup(Operation *op, EvalScope &scope);
  /// Fills the entry for `op` or for an equivalent operation, creating one if
  /// none exists. Only candidates appended since `lookup` are compared.
  CacheEntry *insert(Operation *op, const CacheLookup &lookup,
                     ArrayRef<Answer> answers, EvalScope &scope);
  void fill(CacheEntry *entry, ArrayRef<Answer> answers);
  /// Returns the first entry in `key`'s bucket, starting at index `next`, that
  /// belongs to `op` or to an operation equivalent to it. `next` is left at the
  /// match, or at the bucket size on a miss.
  CacheEntry *findEquivalent(EvaluableOpInterface op, llvm::hash_code key,
                             size_t &next, EvalScope &scope);

  EvalValueStorageAllocator allocator;
  DenseMap<llvm::hash_code, SmallVector<CacheEntry *>> entries;
  llvm::BumpPtrAllocator entryAllocator;
};

} // namespace mlir::interpreter

#endif
