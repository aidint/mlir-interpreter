#ifndef INTERPRETER_EVALCACHE_H
#define INTERPRETER_EVALCACHE_H

#include "interpreter/EvalArena.h"
#include "interpreter/EvalContext.h"
#include "interpreter/EvalValue.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"

#include <optional>

namespace mlir::interpreter {

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

} // namespace mlir::interpreter

#endif
