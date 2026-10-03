#ifndef INTERPRETER_EVALCACHE_H
#define INTERPRETER_EVALCACHE_H

#include "interpreter/EvalContext.h"
#include "interpreter/EvalResult.h"
#include "interpreter/EvalValueStorageAllocator.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <variant>

namespace mlir {
class Operation;
}

namespace mlir::interpreter {

class Engine;

struct CacheEntry {
  Operation *op;
  SmallVector<std::optional<Answer>> results;
};

using CacheResult = std::variant<CacheEntry *, llvm::hash_code, EvalStatus>;

class EvalCache {
public:
  explicit EvalCache(EvalContext &ctx) : allocator(ctx, true) {}

  void beginQuery() { allocator.beginRetaining(); }
  void clear() {
    entries.clear();
    ownedEntries.clear();
    allocator.beginRetaining();
  }

  CacheResult lookup(Operation *op, Engine &engine);
  EvalResult<CacheEntry *> insert(Operation *op, ArrayRef<Answer> answers,
                                  Engine &engine);
  EvalResult<CacheEntry *> insert(Operation *op, llvm::hash_code key,
                                  ArrayRef<Answer> answers, Engine &engine);

private:
  EvalResult<CacheEntry *> findEquivalent(Operation *op, llvm::hash_code key,
                                          Engine &engine);

  EvalValueStorageAllocator allocator;
  DenseMap<llvm::hash_code, SmallVector<CacheEntry *>> entries;
  SmallVector<std::unique_ptr<CacheEntry>> ownedEntries;
};

} // namespace mlir::interpreter

#endif
