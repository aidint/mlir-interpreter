#ifndef INTERPRETER_EVALVALUEINTERFACES_H
#define INTERPRETER_EVALVALUEINTERFACES_H

#include "interpreter/EvalValue.h"

#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Casting.h"

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

} // namespace mlir::interpreter

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::interpreter::EquatableEvalValueInterface)

#endif
