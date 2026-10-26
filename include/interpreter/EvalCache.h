#ifndef INTERPRETER_EVALCACHE_H
#define INTERPRETER_EVALCACHE_H

#include "interpreter/Answer.h"
#include "interpreter/EvalContext.h"
#include "interpreter/EvalResult.h"
#include "interpreter/EvalValue.h"
#include "interpreter/EvalValueStorageAllocator.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"

#include <cassert>
#include <optional>
#include <type_traits>
#include <utility>

namespace mlir {
class Operation;
}

namespace mlir::interpreter {

class Engine;
class EvalScope;
class EvaluableOpInterface;

namespace detail {

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
/// are bucketed by `getHash` and matched with `isEquivalentTo`, so equivalent
/// operations share one entry. Only `Engine` fills it, inside a query, because
/// filling copies values out of the query allocator and dedups them by
/// address, which is only valid while that query runs. Constant-like ops skip
/// the entries: their value depends only on their uniqued attribute and type,
/// so those key a separate map.
class EvalCache {
public:
  void clear() {
    entries.clear();
    entryAllocator.Reset();
    constants.clear();
    allocator.beginRetaining();
  }

private:
  friend class interpreter::EvalContext;
  friend class interpreter::Engine;

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
  /// Returns the cached value of a constant with `attr` and `type`, setting
  /// `hit` to whether one is stored. A hit may return null, which means the
  /// constant has no value.
  EvalValue lookupConstant(Attribute attr, Type type, bool &hit) const {
    auto it = constants.find({attr, type});
    hit = it != constants.end();
    return hit ? it->second : EvalValue();
  }
  /// Stores `value`, or null for no value, as the value of a constant with
  /// `attr` and `type`.
  void insertConstant(Attribute attr, Type type, EvalValue value) {
    assert((!value || value.isCached()) &&
           "constants must be converted into the cache");
    constants.try_emplace({attr, type}, value);
  }

  EvalValueStorageAllocator allocator;
  DenseMap<llvm::hash_code, SmallVector<CacheEntry *>> entries;
  llvm::BumpPtrAllocator entryAllocator;
  /// The values of cacheable constants, keyed by their attribute and type.
  DenseMap<std::pair<Attribute, Type>, EvalValue> constants;
};

} // namespace detail
} // namespace mlir::interpreter

#endif
