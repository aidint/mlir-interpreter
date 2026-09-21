#ifndef INTERPRETER_EVALVALUESUPPORT_H
#define INTERPRETER_EVALVALUESUPPORT_H

#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/TypeID.h"

#include <utility>

namespace mlir::interpreter {

class AbstractEvalValue;
class EvalValueStorageAllocator;
class EvalContext;

namespace detail {

class alignas(8) EvalValueStorage {
public:
  const AbstractEvalValue &getAbstractEvalValue() const {
    return *abstractEvalValue;
  }

private:
  const AbstractEvalValue *abstractEvalValue = nullptr;
  friend class ::mlir::interpreter::EvalValueStorageAllocator;
};

} // namespace detail

class AbstractEvalValue {
public:
  /// Copies `storage` into `allocator` and returns the new storage.
  /// Registration binds this to `EvalValueStorageAllocator::cloneStorage`
  /// instantiated for the value class's storage type, capturing the concrete
  /// type while it is still known, so an allocator can copy a value it only
  /// holds as an `EvalValueStorage *`.
  using CloneFn = detail::EvalValueStorage *(*)(
      EvalValueStorageAllocator &, const detail::EvalValueStorage &);

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
  clone(EvalValueStorageAllocator &allocator,
        const detail::EvalValueStorage &storage) const {
    return cloneFn(allocator, storage);
  }

  TypeID typeID;
  ::mlir::detail::InterfaceMap interfaceMap;
  CloneFn cloneFn;
  friend class EvalValueStorageAllocator;
  friend class EvalContext;
};

} // namespace mlir::interpreter

#endif
