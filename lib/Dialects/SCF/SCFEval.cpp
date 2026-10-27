#include "interpreter/Dialects/SCF/SCFEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"

#include "llvm/ADT/APInt.h"

namespace mlir::interpreter {
namespace {

const APInt *getInt(const Answer &answer) {
  const auto &value = answer.getValue();
  return value ? &cast<IntEvalValue>(value).getValue() : nullptr;
}

/// This class implements the evaluation of `scf.for`. Each iteration binds the
/// induction variable and the iteration arguments, and queries the
/// `scf.yield` operands for the next one. Results are unknown when a bound,
/// the step or an initial value is unknown, or when the loop is undefined.
struct ForOpEval : EvaluableOpInterface::ExternalModel<ForOpEval, scf::ForOp> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    auto forOp = cast<scf::ForOp>(op);
    SmallVector<Answer> unknown(op->getNumResults(), Answer::unknown());

    SmallVector<APInt, 3> control;
    for (Value operand :
         {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()}) {
      Answer answer = scope.query(operand);
      if (!answer.completed())
        return answer.propagate(op);
      const APInt *value = getInt(answer);
      if (!value)
        return unknown;
      control.push_back(*value);
    }
    SmallVector<Answer> carried;
    for (Value init : forOp.getInitArgs()) {
      Answer answer = scope.query(init);
      if (!answer.completed())
        return answer.propagate(op);
      if (answer.isUnknown())
        return unknown;
      carried.push_back(answer);
    }

    bool isUnsigned = forOp.getUnsignedCmp();
    const APInt &ub = control[1], &step = control[2];
    // A step that isn't strictly positive is undefined.
    if (isUnsigned ? step.isZero() : !step.isStrictlyPositive())
      return unknown;
    for (APInt iv = control[0]; isUnsigned ? iv.ult(ub) : iv.slt(ub);) {
      SmallVector<Answer> args{IntEvalValue::get(scope.getAllocator(), iv)};
      llvm::append_range(args, carried);
      // The binding lasts until the end of this iteration.
      BlockBinding binding = scope.bind(forOp.getBody(), args);
      if (!binding)
        return Answer(EvalStatus::Exhausted).propagate(op);
      for (auto [slot, operand] :
           llvm::zip(carried, forOp.getYieldedValues())) {
        slot = scope.query(operand);
        if (!slot.completed())
          return slot.propagate(op);
      }
      bool overflow;
      iv = isUnsigned ? iv.uadd_ov(step, overflow) : iv.sadd_ov(step, overflow);
      // The value past the last iteration must fit the type, or the loop is
      // undefined.
      if (overflow)
        return unknown;
    }
    return carried;
  }

  /// Any nested op may depend on the induction variable or the iteration
  /// arguments, so none is stored.
  bool isNestedOpCacheable(Operation *op, Operation *nested) const {
    return false;
  }
};

} // namespace

void registerSCFEvalExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *) {
    scf::ForOp::attachInterface<ForOpEval>(*ctx);
  });
}

} // namespace mlir::interpreter
