#include "interpreter/EvalContext.h"
#include "interpreter/EvalValueSupport.h"

namespace mlir::interpreter {

EvalContext::EvalContext(MLIRContext *mlirContext)
    : mlirContext(mlirContext) {}

EvalContext::~EvalContext() = default;

void EvalContext::insertAbstractEvalValue(
    TypeID typeID, ::mlir::detail::InterfaceMap &&interfaceMap, CloneFn clone) {
  abstractEvalValues.try_emplace(
      typeID, std::unique_ptr<AbstractEvalValue>(new AbstractEvalValue(
                  typeID, std::move(interfaceMap), clone)));
}

} // namespace mlir::interpreter
