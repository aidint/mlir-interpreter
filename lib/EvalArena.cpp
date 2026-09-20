#include "interpreter/EvalArena.h"
#include "interpreter/EvalValue.h"

namespace mlir::interpreter {

EvalValue EvalArena::retain(EvalValue value) {
  if (value.isCached())
    return value;
  return EvalValue(retainStorage(value.getImpl()), createsCachedValues());
}

detail::EvalValueStorage *
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
