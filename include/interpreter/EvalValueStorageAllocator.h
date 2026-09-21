#ifndef INTERPRETER_EVALVALUESTORAGEALLOCATOR_H
#define INTERPRETER_EVALVALUESTORAGEALLOCATOR_H

#include "interpreter/EvalValueSupport.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"

#include <type_traits>
#include <utility>

namespace mlir::interpreter {

class EvalContext;
class EvalValue;

class EvalValueStorageAllocator {
public:
  EvalValueStorageAllocator(EvalContext &ctx, bool is_cache)
      : ctx(ctx), is_cache(is_cache) {}
  EvalValueStorageAllocator(const EvalValueStorageAllocator &) = delete;
  EvalValueStorageAllocator &
  operator=(const EvalValueStorageAllocator &) = delete;
  ~EvalValueStorageAllocator() {
    for (auto &[ptr, destroy] : llvm::reverse(destructors))
      destroy(ptr);
  }

  EvalContext &getContext() const { return ctx; }
  bool isCacheAllocator() const { return is_cache; }
  void beginRetaining() { retained.clear(); }

  template <typename StorageT, typename... Args>
  StorageT *allocate(const AbstractEvalValue &abstractEvalValue,
                     Args &&...args) {
    static_assert(
        std::is_convertible_v<StorageT *, detail::EvalValueStorage *>,
        "evaluation value storage must publicly derive from EvalValueStorage");
    auto *storage = new (allocator.Allocate<StorageT>())
        StorageT(std::forward<Args>(args)...);
    storage->abstractEvalValue = &abstractEvalValue;
    if constexpr (!std::is_trivially_destructible_v<StorageT>)
      destructors.emplace_back(
          storage, [](void *p) { static_cast<StorageT *>(p)->~StorageT(); });
    return storage;
  }

  EvalValue retain(EvalValue value);

private:
  detail::EvalValueStorage *retainStorage(detail::EvalValueStorage *storage);

  EvalContext &ctx;
  bool is_cache;
  llvm::BumpPtrAllocator allocator;
  SmallVector<std::pair<void *, void (*)(void *)>> destructors;
  DenseMap<detail::EvalValueStorage *, detail::EvalValueStorage *> retained;
};

class EvalSession : public EvalValueStorageAllocator {
public:
  explicit EvalSession(EvalContext &ctx)
      : EvalValueStorageAllocator(ctx, false) {}
};

} // namespace mlir::interpreter

#endif
