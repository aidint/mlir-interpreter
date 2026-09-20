#ifndef INTERPRETER_EVALARENA_H
#define INTERPRETER_EVALARENA_H

#include "interpreter/AbstractEvalValue.h"

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

class EvalArena {
public:
  EvalArena(EvalContext &ctx, bool cached) : ctx(ctx), cached(cached) {}
  EvalArena(const EvalArena &) = delete;
  EvalArena &operator=(const EvalArena &) = delete;
  ~EvalArena() {
    for (auto &[ptr, destroy] : llvm::reverse(destructors))
      destroy(ptr);
  }

  EvalContext &getContext() const { return ctx; }
  bool createsCachedValues() const { return cached; }
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
  bool cached;
  llvm::BumpPtrAllocator allocator;
  SmallVector<std::pair<void *, void (*)(void *)>> destructors;
  DenseMap<detail::EvalValueStorage *, detail::EvalValueStorage *> retained;
};

class EvalSession : public EvalArena {
public:
  explicit EvalSession(EvalContext &ctx) : EvalArena(ctx, false) {}
};

} // namespace mlir::interpreter

#endif
