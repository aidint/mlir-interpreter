#ifndef INTERPRETER_EVALCACHE_H
#define INTERPRETER_EVALCACHE_H

#include "interpreter/Answer.h"
#include "interpreter/EvalCacheKey.h"
#include "interpreter/EvalContext.h"
#include "interpreter/EvalResult.h"
#include "interpreter/EvalValue.h"
#include "interpreter/EvalValueStorageAllocator.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
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

namespace detail {

static_assert(std::is_trivially_destructible_v<std::optional<Answer>>,
              "result slots are bump allocated and never destroyed");

/// This class represents the outcome of a cache lookup: the key built for the
/// operation, and the result slots of its entry, empty on a miss. The key
/// borrows the live ops and the query's values, so it is only valid while
/// both are.
struct CacheLookup {
  CacheKey key;
  MutableArrayRef<std::optional<Answer>> results;
};

/// This class represents the context cache: the known answers operations
/// computed, kept across queries. Each `EvalContext` owns one, shared by every
/// `Engine` on that context, so an answer one query computes is reused by later
/// queries and other engines, and cached values live as long as the context.
/// Entries are keyed by a `CacheKey`, the op's structure plus the answers of
/// its inputs, so operations with equal keys share one entry. A key holds no
/// reference to the live IR, so an entry outlives the op that filled it. Only
/// `Engine` fills it, inside a query, because filling copies values out of the
/// query allocator and dedups them by address, which is only valid while that
/// query runs. Constant-like ops skip the entries: their value depends only on
/// their uniqued attribute and type, so those key a separate map.
class EvalCache {
public:
  EvalCache(const EvalCache &) = delete;
  EvalCache &operator=(const EvalCache &) = delete;
  ~EvalCache() { destroyKeys(); }

  void clear() {
    destroyKeys();
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

  /// Builds the key of `op` in `scratch`, which evaluates its inputs, and finds
  /// its entry. A signal from an input is returned, and nothing is looked up.
  EvalResult<CacheLookup> lookup(Operation *op, EvalScope &scope,
                                 llvm::BumpPtrAllocator &scratch);
  /// Fills the entry for `key`, adding one that owns a copy of `key` if none
  /// exists, and returns its result slots.
  MutableArrayRef<std::optional<Answer>> insert(const CacheKey &key,
                                                ArrayRef<Answer> answers);
  void fill(MutableArrayRef<std::optional<Answer>> results,
            ArrayRef<Answer> answers);
  void destroyKeys();
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
  /// The result slots of each entry. Keys, slots and their arrays live in
  /// `entryAllocator`.
  DenseMap<CacheKey, MutableArrayRef<std::optional<Answer>>> entries;
  llvm::BumpPtrAllocator entryAllocator;
  /// The values of cacheable constants, keyed by their attribute and type.
  DenseMap<std::pair<Attribute, Type>, EvalValue> constants;
};

} // namespace detail
} // namespace mlir::interpreter

#endif
