#ifndef INTERPRETER_EVALCACHE_H
#define INTERPRETER_EVALCACHE_H

#include "interpreter/EvalContext.h"
#include "interpreter/EvalValue.h"
#include "interpreter/EvalValueStorageAllocator.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"

#include <memory>
#include <optional>

namespace mlir::interpreter {

struct CacheEntry {
  Operation *op;
  SmallVector<std::optional<Answer>> results;
};

using CacheResult = std::variant<CacheEntry *, llvm::hash_code, AnswerKind>;

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
  CacheKeyResult<CacheEntry *> insert(Operation *op, ArrayRef<Answer> answers,
                                      Engine &engine);
  CacheKeyResult<CacheEntry *> insert(Operation *op, llvm::hash_code key,
                                      ArrayRef<Answer> answers, Engine &engine);

private:
  CacheKeyResult<CacheEntry *>
  findEquivalent(Operation *op, llvm::hash_code key, Engine &engine);

  EvalValueStorageAllocator allocator;
  DenseMap<llvm::hash_code, SmallVector<CacheEntry *>> entries;
  SmallVector<std::unique_ptr<CacheEntry>> ownedEntries;
};

} // namespace mlir::interpreter

#endif
