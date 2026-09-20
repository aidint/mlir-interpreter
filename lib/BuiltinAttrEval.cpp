#include "interpreter/BuiltinAttrEval.h"
#include "interpreter/BuiltinEvalValues.h"
#include "interpreter/EvaluableAttrInterface.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir::interpreter {
namespace {

struct IntegerAttrEval
    : EvaluableAttrInterface::ExternalModel<IntegerAttrEval, IntegerAttr> {
  std::optional<EvalValue> toEvalValue(Attribute attr,
                                       EvalSession &session) const {
    return IntEvalValue::get(session, cast<IntegerAttr>(attr).getValue());
  }
};

} // namespace

void registerBuiltinAttrEvalExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *) {
    IntegerAttr::attachInterface<IntegerAttrEval>(*ctx);
  });
}

} // namespace mlir::interpreter
