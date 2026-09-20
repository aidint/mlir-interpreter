#ifndef INTERPRETER_DIALECTS_BUILTIN_BUILTINEVALVALUES_H
#define INTERPRETER_DIALECTS_BUILTIN_BUILTINEVALVALUES_H

#include "interpreter/EvalContext.h"
#include "interpreter/EvalValue.h"
#include "interpreter/EvalValueInterfaces.h"

#include "llvm/ADT/APInt.h"

namespace mlir::interpreter {

namespace detail {

struct IntEvalValueStorage : public EvalValueStorage {
  IntEvalValueStorage(APInt value) : value(std::move(value)) {}
  APInt value;
};

} // namespace detail

class IntEvalValue
    : public EvalValueBase<IntEvalValue, detail::IntEvalValueStorage,
                           EquatableEvalValueInterface::Trait> {
public:
  using Base::Base;

  const APInt &getValue() const { return getImpl()->value; }

  bool isEqual(IntEvalValue other) const {
    const APInt &rhs = other.getValue();
    return getValue().getBitWidth() == rhs.getBitWidth() && getValue() == rhs;
  }

  llvm::hash_code hash() const { return llvm::hash_value(getValue()); }
};

void registerBuiltinEvalValues(EvalContext &ctx);

} // namespace mlir::interpreter

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::interpreter::IntEvalValue)

#endif
