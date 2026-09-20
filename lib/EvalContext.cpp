#include "interpreter/EvalValue.h"

namespace mlir::interpreter {

EvalContext::EvalContext(MLIRContext *mlirContext)
    : mlirContext(mlirContext), cache(std::make_unique<EvalCache>(*this)) {}

EvalContext::~EvalContext() = default;

void EvalContext::clearCache() { cache->clear(); }

} // namespace mlir::interpreter
