#include "mlir-interpreter/BuiltinAttrEval.h"
#include "mlir-interpreter/EvaluableAttrInterface.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir::interpreter {
namespace {

struct IntegerAttrEval
    : EvaluableAttrInterface::ExternalModel<IntegerAttrEval, IntegerAttr> {
  std::optional<InterpreterValue> toInterpreterValue(Attribute attr) const {
    return InterpreterValue{cast<IntegerAttr>(attr).getValue()};
  }
};

} // namespace

void registerBuiltinAttrEvalExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *) {
    IntegerAttr::attachInterface<IntegerAttrEval>(*ctx);
  });
}

} // namespace mlir::interpreter
