#ifndef INTERPRETER_EVALCONTEXT_H
#define INTERPRETER_EVALCONTEXT_H

#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"

#include <memory>
#include <type_traits>
#include <utility>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::interpreter {

class AbstractEvalValue;
class EvalArena;
class EvalCache;
class EvalContext;
class EvalValue;

namespace detail {

class alignas(8) EvalValueStorage {
public:
  const AbstractEvalValue &getAbstractEvalValue() const {
    return *abstractEvalValue;
  }

private:
  const AbstractEvalValue *abstractEvalValue = nullptr;
  friend class ::mlir::interpreter::EvalArena;
};

} // namespace detail

class AbstractEvalValue {
public:
  using CloneFn = detail::EvalValueStorage *(*)(
      EvalArena &, const detail::EvalValueStorage &);

  TypeID getTypeID() const { return typeID; }

  template <typename T> typename T::Concept *getInterface() const {
    return interfaceMap.lookup<T>();
  }

  bool hasInterface(TypeID interfaceID) const {
    return interfaceMap.contains(interfaceID);
  }

  template <typename... Models> void attachInterface() {
    interfaceMap.insertModels<Models...>();
  }

private:
  AbstractEvalValue(TypeID typeID, ::mlir::detail::InterfaceMap &&interfaceMap,
                    CloneFn clone)
      : typeID(typeID), interfaceMap(std::move(interfaceMap)), cloneFn(clone) {}

  detail::EvalValueStorage *
  clone(EvalArena &arena, const detail::EvalValueStorage &storage) const {
    return cloneFn(arena, storage);
  }

  TypeID typeID;
  ::mlir::detail::InterfaceMap interfaceMap;
  CloneFn cloneFn;
  friend class EvalArena;
  friend class EvalContext;
};

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
  StorageT *allocate(const AbstractEvalValue &abstract, Args &&...args) {
    static_assert(
        std::is_convertible_v<StorageT *, detail::EvalValueStorage *>,
        "evaluation value storage must publicly derive from EvalValueStorage");
    auto *storage = new (allocator.Allocate<StorageT>())
        StorageT(std::forward<Args>(args)...);
    storage->abstractEvalValue = &abstract;
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

class EvalContext {
public:
  explicit EvalContext(MLIRContext *mlirContext);
  EvalContext(const EvalContext &) = delete;
  EvalContext &operator=(const EvalContext &) = delete;
  ~EvalContext();

  MLIRContext *getMLIRContext() const { return mlirContext; }

  template <typename T> void registerEvalValue();

  template <typename T> bool isRegistered() const {
    return abstracts.contains(TypeID::get<T>());
  }

  template <typename T> const AbstractEvalValue &getAbstract() const {
    auto it = abstracts.find(TypeID::get<T>());
    assert(it != abstracts.end() && "value class not registered in context");
    return *it->second;
  }

  template <typename T, typename... Models> void attachInterface() {
    auto it = abstracts.find(TypeID::get<T>());
    assert(it != abstracts.end() && "value class not registered in context");
    it->second->template attachInterface<Models...>();
  }

  EvalCache &getCache() { return *cache; }
  const EvalCache &getCache() const { return *cache; }
  void clearCache();

private:
  MLIRContext *mlirContext;
  DenseMap<TypeID, std::unique_ptr<AbstractEvalValue>> abstracts;
  std::unique_ptr<EvalCache> cache;
};

} // namespace mlir::interpreter

#endif
