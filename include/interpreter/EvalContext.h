#ifndef INTERPRETER_EVALCONTEXT_H
#define INTERPRETER_EVALCONTEXT_H

#include "interpreter/EvalValueStorageAllocator.h"

#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/DenseMap.h"

#include <cassert>
#include <memory>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::interpreter {

class AbstractEvalValue;
class EvalValueStorageAllocator;
class EvalCache;

namespace detail {
class EvalValueStorage;
} // namespace detail

class EvalContext {
public:
  explicit EvalContext(MLIRContext *mlirContext);
  EvalContext(const EvalContext &) = delete;
  EvalContext &operator=(const EvalContext &) = delete;
  ~EvalContext();

  MLIRContext *getMLIRContext() const { return mlirContext; }

  template <typename... Ts> void registerEvalValues() {
    (registerAbstractEvalValue<Ts>(), ...);
  }

  template <typename T> bool isRegistered() const {
    return abstractEvalValues.contains(TypeID::get<T>());
  }

  template <typename T> const AbstractEvalValue &getAbstractEvalValue() const {
    auto it = abstractEvalValues.find(TypeID::get<T>());
    assert(it != abstractEvalValues.end() &&
           "value class not registered in context");
    return *it->second;
  }

  template <typename T, typename... Models> void attachInterface() {
    auto it = abstractEvalValues.find(TypeID::get<T>());
    assert(it != abstractEvalValues.end() &&
           "value class not registered in context");
    it->second->template attachInterface<Models...>();
  }

  EvalCache &getCache() { return *cache; }
  const EvalCache &getCache() const { return *cache; }
  void clearCache();

private:
  using CloneFn = detail::EvalValueStorage *(*)(
      EvalValueStorageAllocator &, const detail::EvalValueStorage &);

  template <typename T> void registerAbstractEvalValue() {
    TypeID typeID = TypeID::get<T>();
    if (abstractEvalValues.contains(typeID))
      return;
    using StorageT = typename T::ImplType;
    auto clone = +[](EvalValueStorageAllocator &allocator,
                     const detail::EvalValueStorage &storage) {
      return static_cast<detail::EvalValueStorage *>(
          allocator.cloneStorage<StorageT>(
              static_cast<const StorageT *>(&storage)));
    };
    insertAbstractEvalValue(typeID, T::getInterfaceMap(), clone);
  }

  void insertAbstractEvalValue(TypeID typeID,
                               ::mlir::detail::InterfaceMap &&interfaceMap,
                               CloneFn clone);

  MLIRContext *mlirContext;
  DenseMap<TypeID, std::unique_ptr<AbstractEvalValue>> abstractEvalValues;
  std::unique_ptr<EvalCache> cache;
};

} // namespace mlir::interpreter

#endif
