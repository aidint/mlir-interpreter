#include "interpreter/Dialects/Builtin/BuiltinAttrEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/EvaluableAttrInterface.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir::interpreter {
namespace {

struct IntegerAttrEval
    : EvaluableAttrInterface::ExternalModel<IntegerAttrEval, IntegerAttr> {
  std::optional<EvalValue>
  toEvalValue(Attribute attr, EvalValueStorageAllocator &allocator) const {
    return IntEvalValue::get(allocator, cast<IntegerAttr>(attr).getValue());
  }
};

} // namespace

void registerBuiltinAttrEvalExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *) {
    IntegerAttr::attachInterface<IntegerAttrEval>(*ctx);
  });
}

} // namespace mlir::interpreter
