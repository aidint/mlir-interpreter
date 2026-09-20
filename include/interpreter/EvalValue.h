#ifndef INTERPRETER_EVALVALUE_H
#define INTERPRETER_EVALVALUE_H

#include "interpreter/EvalContext.h"

#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Casting.h"

#include <optional>
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
    return ConcreteT(arena.allocate<StorageT>(ctx.getAbstract<ConcreteT>(),
                                              std::forward<Args>(args)...),
                     arena.createsCachedValues());
  }

protected:
  StorageT *getImpl() const {
    return static_cast<StorageT *>(EvalValue::getImpl());
  }
};

inline EvalValue EvalArena::retain(EvalValue value) {
  if (value.isCached())
    return value;
  return EvalValue(retainStorage(value.getImpl()), createsCachedValues());
}

class EvalCache {
public:
  explicit EvalCache(EvalContext &ctx) : arena(ctx, true) {}

  void beginQuery() { arena.beginRetaining(); }
  void clear() {
    values.clear();
    arena.beginRetaining();
  }

  bool lookup(Value value, std::optional<EvalValue> &result) const {
    auto it = values.find(value);
    if (it == values.end())
      return false;
    result = it->second;
    return true;
  }

  void insert(Value value, std::optional<EvalValue> result) {
    if (values.contains(value))
      return;
    if (result)
      result = arena.retain(*result);
    values.try_emplace(value, result);
  }

private:
  EvalArena arena;
  DenseMap<Value, std::optional<EvalValue>> values;
};

class EvalBasket : public EvalArena {
public:
  explicit EvalBasket(EvalContext &ctx)
      : EvalArena(ctx, false), cache(ctx.getCache()) {}

  void beginQuery() { beginRetaining(); }

  EvalValue retain(EvalValue value) { return EvalArena::retain(value); }

private:
  EvalCache &cache;
};

inline detail::EvalValueStorage *
EvalArena::retainStorage(detail::EvalValueStorage *storage) {
  auto [it, inserted] = retained.try_emplace(storage);
  if (!inserted)
    return it->second;
  detail::EvalValueStorage *copy =
      storage->getAbstractEvalValue().clone(*this, *storage);
  it->second = copy;
  retained.try_emplace(copy, copy);
  return copy;
}

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

namespace mlir::interpreter {

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

class EquatableEvalValueInterface;

struct EquatableEvalValueInterfaceTraits {
  struct Concept {
    bool (*isEqual)(const Concept *impl, EvalValue lhs, EvalValue rhs);
    llvm::hash_code (*hash)(const Concept *impl, EvalValue value);
  };

  template <typename ConcreteValue> struct Model : public Concept {
    using Interface = EquatableEvalValueInterface;
    Model() : Concept{isEqual, hash} {}
    static bool isEqual(const Concept *, EvalValue lhs, EvalValue rhs) {
      return llvm::cast<ConcreteValue>(lhs).isEqual(
          llvm::cast<ConcreteValue>(rhs));
    }
    static llvm::hash_code hash(const Concept *, EvalValue value) {
      return llvm::cast<ConcreteValue>(value).hash();
    }
  };

  template <typename ConcreteModel> struct FallbackModel : public Concept {
    using Interface = EquatableEvalValueInterface;
    FallbackModel() : Concept{isEqual, hash} {}
    static bool isEqual(const Concept *impl, EvalValue lhs, EvalValue rhs) {
      return static_cast<const ConcreteModel *>(impl)->isEqual(lhs, rhs);
    }
    static llvm::hash_code hash(const Concept *impl, EvalValue value) {
      return static_cast<const ConcreteModel *>(impl)->hash(value);
    }
  };

  template <typename ConcreteModel, typename ConcreteValue>
  struct ExternalModel : public FallbackModel<ConcreteModel> {};
};

class EquatableEvalValueInterface
    : public EvalValueInterface<EquatableEvalValueInterface,
                                EquatableEvalValueInterfaceTraits> {
public:
  using Base::Base;

  bool isEqual(EvalValue other) const {
    if (*this == other)
      return true;
    if (!other || getTypeID() != other.getTypeID())
      return false;
    return getImpl()->isEqual(getImpl(), *this, other);
  }

  llvm::hash_code hash() const { return getImpl()->hash(getImpl(), *this); }
};

inline bool isEqual(EvalValue lhs, EvalValue rhs) {
  if (lhs == rhs)
    return true;
  auto equatable = llvm::dyn_cast_if_present<EquatableEvalValueInterface>(lhs);
  return equatable && equatable.isEqual(rhs);
}

template <typename T> void EvalContext::registerEvalValue() {
  TypeID typeID = TypeID::get<T>();
  if (abstracts.contains(typeID))
    return;
  using StorageT = typename T::ImplType;
  auto clone = +[](EvalArena &arena, const detail::EvalValueStorage &storage) {
    auto &concrete = static_cast<const StorageT &>(storage);
    return static_cast<detail::EvalValueStorage *>(
        T::cloneStorage(arena, concrete));
  };
  abstracts.try_emplace(
      typeID, std::unique_ptr<AbstractEvalValue>(
                  new AbstractEvalValue(typeID, T::getInterfaceMap(), clone)));
}

} // namespace mlir::interpreter

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::interpreter::EquatableEvalValueInterface)

#endif
