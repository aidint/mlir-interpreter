#ifndef INTERPRETER_EVALVALUESUPPORT_H
#define INTERPRETER_EVALVALUESUPPORT_H

#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/Hashing.h"

#include <utility>

namespace mlir::interpreter {

class AbstractEvalValue;
class EvalValue;
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
  /// Hashes a value, and compares two values of this class. Registration binds
  /// them to the class's `hash` and `isEqual`, or to its identity when the
  /// class defines neither; `EvalValue::hash` and `EvalValue::isEqual`
  /// dispatch to them.
  using HashFn = llvm::hash_code (*)(EvalValue);
  using IsEqualFn = bool (*)(EvalValue, EvalValue);

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
                    CloneFn clone, HashFn hash, IsEqualFn isEqual)
      : typeID(typeID), interfaceMap(std::move(interfaceMap)), cloneFn(clone),
        hashFn(hash), isEqualFn(isEqual) {}

  detail::EvalValueStorage *
  clone(EvalValueStorageAllocator &allocator,
        const detail::EvalValueStorage &storage) const {
    return cloneFn(allocator, storage);
  }

  TypeID typeID;
  ::mlir::detail::InterfaceMap interfaceMap;
  CloneFn cloneFn;
  HashFn hashFn;
  IsEqualFn isEqualFn;
  friend class EvalValue;
  friend class EvalValueStorageAllocator;
  friend class EvalContext;
};

} // namespace mlir::interpreter

#endif
