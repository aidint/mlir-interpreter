#include "interpreter/Dialects/Arith/ArithEval.h"
#include "interpreter/Dialects/Builtin/BuiltinAttrEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Engine.h"
#include "interpreter/EvalContext.h"
#include "interpreter/EvalValue.h"
#include "interpreter/EvaluableOpInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;
using namespace mlir::interpreter;

namespace {

int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      llvm::errs() << "FAIL " << __LINE__ << ": " << #cond << "\n";            \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

unsigned liveTracked = 0;

struct TrackedEvalValueStorage
    : public mlir::interpreter::detail::EvalValueStorage {
  TrackedEvalValueStorage(APInt value) : value(std::move(value)) {
    ++liveTracked;
  }
  TrackedEvalValueStorage(const TrackedEvalValueStorage &other)
      : value(other.value) {
    ++liveTracked;
  }
  ~TrackedEvalValueStorage() { --liveTracked; }
  APInt value;
};

class TrackedEvalValue
    : public EvalValueBase<TrackedEvalValue, TrackedEvalValueStorage> {
public:
  using Base::Base;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TrackedEvalValue)

  const APInt &getValue() const { return getImpl()->value; }
};

struct SubIOpEval
    : EvaluableOpInterface::ExternalModel<SubIOpEval, arith::SubIOp> {
  SmallVector<Answer> evaluate(Operation *,
                               ArrayRef<std::optional<EvalValue>> operands,
                               EvalScope &scope) const {
    if (!operands[0] || !operands[1])
      return {{AnswerKind::Unknown, std::nullopt}};
    const APInt &lhs = cast<IntEvalValue>(*operands[0]).getValue();
    const APInt &rhs = cast<IntEvalValue>(*operands[1]).getValue();
    return {{AnswerKind::Known,
             TrackedEvalValue::get(scope.getSession(), lhs - rhs)}};
  }

  bool isCacheable(Operation *) const { return false; }
};

const char *moduleSource = R"mlir(
func.func @f() -> (i32, i32) {
  %c7 = arith.constant 7 : i32
  %c2 = arith.constant 2 : i32
  %c9 = arith.constant 9 : i32
  %c3 = arith.constant 3 : i32
  %a = arith.subi %c7, %c2 : i32
  %b = arith.subi %c9, %c3 : i32
  return %a, %b : i32, i32
}
)mlir";

const APInt *getTracked(const Answer &answer) {
  if (!answer.value || !*answer.value)
    return nullptr;
  auto tracked = dyn_cast<TrackedEvalValue>(*answer.value);
  return tracked ? &tracked.getValue() : nullptr;
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect>();
  registerArithEvalExternalModels(registry);
  registerBuiltinAttrEvalExternalModels(registry);
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *) {
    arith::SubIOp::attachInterface<SubIOpEval>(*ctx);
  });

  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleSource, &context);
  if (!module) {
    llvm::errs() << "failed to parse the test module\n";
    return 1;
  }

  SmallVector<Value> uncached;
  SmallVector<Value> constants;
  module->walk([&](Operation *op) {
    if (isa<arith::SubIOp>(op))
      uncached.push_back(op->getResult(0));
    else if (isa<arith::ConstantOp>(op))
      constants.push_back(op->getResult(0));
  });
  CHECK(uncached.size() == 2);
  CHECK(constants.size() == 4);
  if (failures)
    return 1;

  EvalContext evalContext(&context);
  registerBuiltinEvalValues(evalContext);
  evalContext.registerEvalValues<TrackedEvalValue>();

  {
    Engine engine(evalContext, 100);

    Answer first = engine.query(uncached[0]);
    CHECK(first.kind == AnswerKind::Known);
    CHECK(first.value && !first.value->isCached());
    CHECK(getTracked(first) && *getTracked(first) == 5);
    CHECK(liveTracked == 1);

    Answer second = engine.query(uncached[1]);
    CHECK(second.kind == AnswerKind::Known);
    CHECK(getTracked(second) && *getTracked(second) == 6);
    CHECK(liveTracked == 2);

    CHECK(getTracked(first) && *getTracked(first) == 5);
    CHECK(first.value->getImpl() != second.value->getImpl());

    Answer constant = engine.query(constants[0]);
    CHECK(constant.kind == AnswerKind::Known);
    CHECK(constant.value && constant.value->isCached());
    CHECK(cast<IntEvalValue>(*constant.value).getValue() == 7);
    CHECK(liveTracked == 2);

    Answer again = engine.query(uncached[0]);
    CHECK(getTracked(again) && *getTracked(again) == 5);
    CHECK(again.value->getImpl() != first.value->getImpl());
    CHECK(liveTracked == 3);
  }
  CHECK(liveTracked == 0);

  evalContext.clearCache();
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
