#ifndef INTERPRETER_ABSTRACTEVALVALUE_H
#define INTERPRETER_ABSTRACTEVALVALUE_H

#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/TypeID.h"

#include <utility>

namespace mlir::interpreter {

class AbstractEvalValue;
class EvalArena;
class EvalContext;

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

} // namespace mlir::interpreter

#endif
