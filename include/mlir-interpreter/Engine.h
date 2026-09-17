#ifndef MLIR_INTERPRETER_ENGINE_H
#define MLIR_INTERPRETER_ENGINE_H

#include "mlir-interpreter/EvaluableOpInterface.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::interpreter {

class Engine {
public:
  explicit Engine(uint64_t budget);

  void registerType(TypeID type);

  Answer query(Value value);
  FailureOr<func::FuncOp> specialize(func::CallOp call,
                                     ArrayRef<std::optional<InterpreterValue>> args);

private:
  Answer evaluate(Value value);

  uint64_t budget;
  uint64_t remaining = 0;
  unsigned depth = 0;
};

StringRef stringifyAnswerKind(AnswerKind kind);

} // namespace mlir::interpreter

#endif
