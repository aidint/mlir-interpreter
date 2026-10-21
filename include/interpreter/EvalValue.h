#ifndef INTERPRETER_EVALVALUE_H
#define INTERPRETER_EVALVALUE_H

#include "interpreter/EvalContext.h"
#include "interpreter/EvalValueStorageAllocator.h"
#include "interpreter/EvalValueSupport.h"

#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <type_traits>
#include <utility>

namespace mlir::interpreter {

/// This class represents an evaluated value: a pointer-sized handle to storage
/// owned by a query, cache or answer allocator. `operator==` compares handles,
/// while `isEqual` and `hash` compare contents as the value's class defines.
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

  /// Returns a hash that is equal for values `isEqual` matches.
  llvm::hash_code hash() const { return getAbstractEvalValue().hashFn(*this); }

  /// Returns whether this value equals `other`. A value equals itself, values
  /// of different classes never match, and the rest compare with their class's
  /// `isEqual`. Unlike `operator==`, which compares handles, this may match
  /// distinct instances.
  bool isEqual(EvalValue other) const {
    if (getImpl() == other.getImpl())
      return true;
    assert(*this && other && "cannot compare a null value");
    if (getTypeID() != other.getTypeID())
      return false;
    return getAbstractEvalValue().isEqualFn(*this, other);
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

  /// Returns the function `EvalValue::hash` dispatches to: `ConcreteT::hash`,
  /// or the instance's identity when the class keeps the default.
  static AbstractEvalValue::HashFn getHashFn() {
    if constexpr (definesEquality())
      return [](EvalValue value) { return cast<ConcreteT>(value).hash(); };
    else
      return [](EvalValue value) { return llvm::hash_value(value.getImpl()); };
  }

  /// Returns the function `EvalValue::isEqual` dispatches to once both values
  /// are known to be `ConcreteT`s.
  static AbstractEvalValue::IsEqualFn getIsEqualFn() {
    if constexpr (definesEquality())
      return [](EvalValue lhs, EvalValue rhs) {
        return cast<ConcreteT>(lhs).isEqual(cast<ConcreteT>(rhs));
      };
    else
      return [](EvalValue lhs, EvalValue rhs) {
        return lhs.getImpl() == rhs.getImpl();
      };
  }

  template <typename... Args>
  static ConcreteT get(EvalValueStorageAllocator &allocator, Args &&...args) {
    EvalContext &ctx = allocator.getContext();
    return ConcreteT(
        allocator.allocate<StorageT>(ctx.getAbstractEvalValue<ConcreteT>(),
                                     std::forward<Args>(args)...),
        allocator.isCacheAllocator());
  }

protected:
  StorageT *getImpl() const {
    return static_cast<StorageT *>(EvalValue::getImpl());
  }

private:
  /// Returns whether `ConcreteT` declares its own `hash` and `isEqual` instead
  /// of inheriting `EvalValue`'s, which would dispatch back to it.
  static constexpr bool definesEquality() {
    constexpr bool hash =
        !std::is_same_v<decltype(&ConcreteT::hash), decltype(&EvalValue::hash)>;
    constexpr bool isEqual = !std::is_same_v<decltype(&ConcreteT::isEqual),
                                             decltype(&EvalValue::isEqual)>;
    static_assert(hash == isEqual,
                  "a value class must define both `hash` and `isEqual`, or "
                  "neither");
    return hash;
  }
};

template <typename ConcreteType, template <typename> class TraitType>
class EvalValueTraitBase {};

template <typename ConcreteType, typename Traits>
class EvalValueInterface
    : public ::mlir::detail::Interface<ConcreteType, EvalValue, Traits,
                                       EvalValue, EvalValueTraitBase> {
public:
  using Base = EvalValueInterface<ConcreteType, Traits>;
  using InterfaceBase =
      ::mlir::detail::Interface<ConcreteType, EvalValue, Traits, EvalValue,
                                EvalValueTraitBase>;
  using InterfaceBase::InterfaceBase;

private:
  static typename InterfaceBase::Concept *getInterfaceFor(EvalValue value) {
    return value.getAbstractEvalValue().getInterface<ConcreteType>();
  }

  friend InterfaceBase;
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
