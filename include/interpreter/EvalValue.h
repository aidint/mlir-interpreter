#ifndef INTERPRETER_EVALVALUE_H
#define INTERPRETER_EVALVALUE_H

#include "interpreter/AbstractEvalValue.h"
#include "interpreter/EvalArena.h"
#include "interpreter/EvalContext.h"

#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <type_traits>
#include <utility>

namespace mlir::interpreter {

class EvalValue {
public:
  using ImplType = detail::EvalValueStorage;
  using ImplAndCached = llvm::PointerIntPair<ImplType *, 1, bool>;

  EvalValue() = default;
  EvalValue(ImplType *impl, bool cached = false) : impl(impl, cached) {}

  explicit operator bool() const { return getImpl(); }
  bool operator==(EvalValue other) const {
    return getAsOpaquePointer() == other.getAsOpaquePointer();
  }
  bool operator!=(EvalValue other) const { return !(*this == other); }

  TypeID getTypeID() const { return getAbstractEvalValue().getTypeID(); }
  const AbstractEvalValue &getAbstractEvalValue() const {
    return getImpl()->getAbstractEvalValue();
  }

  template <typename T> bool hasInterface() const {
    return getAbstractEvalValue().hasInterface(TypeID::get<T>());
  }

  bool isCached() const { return impl.getInt(); }
  ImplType *getImpl() const { return impl.getPointer(); }
  const void *getAsOpaquePointer() const { return impl.getOpaqueValue(); }
  static EvalValue getFromOpaquePointer(const void *pointer) {
    EvalValue value;
    value.impl = ImplAndCached::getFromOpaqueValue(const_cast<void *>(pointer));
    return value;
  }

protected:
  ImplAndCached impl;
};

inline llvm::hash_code hash_value(EvalValue value) {
  return llvm::hash_value(value.getAsOpaquePointer());
}

template <typename ConcreteT, typename StorageT,
          template <typename> class... Traits>
class EvalValueBase : public EvalValue {
public:
  using Base = EvalValueBase;
  using ImplType = StorageT;
  using EvalValue::EvalValue;

  EvalValueBase(EvalValue value) : EvalValue(value) {}

  static bool classof(EvalValue value) {
    return value.getTypeID() == TypeID::get<ConcreteT>();
  }

  static ::mlir::detail::InterfaceMap getInterfaceMap() {
    return ::mlir::detail::InterfaceMap::get<Traits<ConcreteT>...>();
  }

  static StorageT *cloneStorage(EvalArena &arena, const StorageT &storage) {
    return arena.allocate<StorageT>(storage.getAbstractEvalValue(), storage);
  }

  template <typename... Args>
  static ConcreteT get(EvalArena &arena, Args &&...args) {
    EvalContext &ctx = arena.getContext();
    return ConcreteT(
        arena.allocate<StorageT>(ctx.getAbstractEvalValue<ConcreteT>(),
                                 std::forward<Args>(args)...),
        arena.createsCachedValues());
  }

protected:
  StorageT *getImpl() const {
    return static_cast<StorageT *>(EvalValue::getImpl());
  }
};

} // namespace mlir::interpreter

namespace llvm {

template <> struct DenseMapInfo<mlir::interpreter::EvalValue> {
  using EvalValue = mlir::interpreter::EvalValue;
  static unsigned getHashValue(EvalValue value) {
    return mlir::interpreter::hash_value(value);
  }
  static bool isEqual(EvalValue lhs, EvalValue rhs) { return lhs == rhs; }
};

template <typename To, typename From>
struct CastInfo<
    To, From,
    std::enable_if_t<std::is_same_v<mlir::interpreter::EvalValue,
                                    std::remove_const_t<From>> ||
                     std::is_base_of_v<mlir::interpreter::EvalValue, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  static inline bool isPossible(mlir::interpreter::EvalValue value) {
    if constexpr (std::is_base_of_v<To, From>)
      return true;
    else
      return To::classof(value);
  }
  static inline To doCast(mlir::interpreter::EvalValue value) {
    return To(value);
  }
};

} // namespace llvm

#endif
