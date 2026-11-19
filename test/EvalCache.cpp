#include "InterpTestDialect.h"
#include "interpreter/Dialects/Builtin/BuiltinAttrEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Dialects/SCF/SCFEval.h"
#include "interpreter/Engine.h"
#include "interpreter/Interfaces/EvaluableAttrInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <map>

using namespace mlir;
using namespace mlir::interpreter;

namespace {

unsigned failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      llvm::errs() << "FAIL " << __LINE__ << ": " << #condition << "\n";       \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

Answer integer(EvalValueStorageAllocator &allocator, int64_t value) {
  return IntEvalValue::get(allocator, APInt(32, value));
}

bool isInteger(const Answer &answer, int64_t value) {
  return answer.completed() && !answer.isUnknown() &&
         cast<IntEvalValue>(answer.getValue()).getValue() == value;
}

struct OpaqueEvalValueStorage
    : public mlir::interpreter::detail::EvalValueStorage {};

/// This class represents a test value that keeps the identity `hash` and
/// `isEqual`, so a key holding one matches only the same instance.
class OpaqueEvalValue
    : public EvalValueBase<OpaqueEvalValue, OpaqueEvalValueStorage> {
public:
  using Base::Base;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpaqueEvalValue)
};

struct Behavior {
  unsigned eligibilityChecks = 0;
  unsigned evaluations = 0;
  std::function<EvalResult<bool>(Operation *, EvalScope &)> cacheable =
      [](Operation *, EvalScope &) { return true; };
  std::function<SmallVector<Answer>(Operation *, EvalScope &)> evaluate =
      [](Operation *op, EvalScope &scope) {
        for (Value operand : op->getOperands()) {
          Answer answer = scope.query(operand);
          if (!answer.completed())
            return answer.propagate(op);
        }
        return SmallVector<Answer>(op->getNumResults(),
                                   integer(scope.getAllocator(), 7));
      };
};

std::map<Operation *, Behavior> behaviors;

/// This class represents a test model whose eligibility and evaluation come
/// from the op's `Behavior`.
struct TestModel
    : EvaluableOpInterface::ExternalModel<TestModel,
                                          UnrealizedConversionCastOp> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    auto &behavior = behaviors.at(op);
    ++behavior.evaluations;
    return behavior.evaluate(op, scope);
  }

  EvalResult<bool> isCacheable(Operation *op, EvalScope &scope) const {
    auto &behavior = behaviors.at(op);
    ++behavior.eligibilityChecks;
    return behavior.cacheable(op, scope);
  }
};

/// This class represents a test model that keeps every default, so whether its
/// op is cached follows the op's side effects.
template <typename OpT>
struct DefaultModel
    : EvaluableOpInterface::ExternalModel<DefaultModel<OpT>, OpT> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    auto &behavior = behaviors[op];
    ++behavior.evaluations;
    return behavior.evaluate(op, scope);
  }
};

void uncacheable(Operation *op) {
  behaviors.at(op).cacheable = [](Operation *, EvalScope &) { return false; };
}

/// How the test model of `#ub.poison` converts it; reset by each `Fixture`.
struct PoisonBehavior {
  bool cacheable = true;
  bool known = true;
  unsigned conversions = 0;
} poison;

/// This class represents a test model for `#ub.poison`, which `ub.poison`
/// folds to whatever its type, so constants with one attribute can differ in
/// type. Its values keep the identity `isEqual`, so a shared value is the
/// same instance.
struct PoisonAttrEval
    : EvaluableAttrInterface::ExternalModel<PoisonAttrEval, ub::PoisonAttr> {
  EvalValue toEvalValue(Attribute, EvalValueStorageAllocator &allocator) const {
    ++poison.conversions;
    if (!poison.known)
      return {};
    return OpaqueEvalValue::get(allocator);
  }

  bool isCacheable(Attribute) const { return poison.cacheable; }
};

DialectRegistry registry() {
  DialectRegistry result;
  result.insert<arith::ArithDialect>();
  registerBuiltinAttrEvalExternalModels(result);
  registerSCFEvalExternalModels(result);
  return result;
}

struct Fixture {
  MLIRContext context{registry()};
  EvalContext evalContext{&context};
  OwningOpRef<ModuleOp> module{ModuleOp::create(UnknownLoc::get(&context))};
  OpBuilder builder{module->getBody(), module->getBody()->begin()};

  Fixture() {
    context
        .loadDialect<arith::ArithDialect, interp_test::InterpTestDialect,
                     memref::MemRefDialect, scf::SCFDialect, ub::UBDialect>();
    UnrealizedConversionCastOp::attachInterface<TestModel>(context);
    ub::PoisonAttr::attachInterface<PoisonAttrEval>(context);
    arith::AddIOp::attachInterface<DefaultModel<arith::AddIOp>>(context);
    memref::LoadOp::attachInterface<DefaultModel<memref::LoadOp>>(context);
    scf::ExecuteRegionOp::attachInterface<DefaultModel<scf::ExecuteRegionOp>>(
        context);
    registerBuiltinEvalValues(evalContext);
    evalContext.registerEvalValues<OpaqueEvalValue>();
  }

  ~Fixture() {
    behaviors.clear();
    poison = {};
  }

  /// Returns an op whose `key` attribute is `key`, so it shares an entry with
  /// any op with the same key, result count and input answers.
  Operation *keyedOp(int64_t key, unsigned results = 1,
                     ValueRange operands = {}) {
    OperationState state(builder.getUnknownLoc(),
                         UnrealizedConversionCastOp::getOperationName());
    state.addTypes(SmallVector<Type>(results, builder.getI32Type()));
    state.addOperands(operands);
    state.addAttribute("key", builder.getI64IntegerAttr(key));
    Operation *result = builder.create(state);
    behaviors.try_emplace(result);
    return result;
  }

  /// Returns an op that shares no entry. Its key is negative, so it differs
  /// from every key a test picks.
  Operation *op(unsigned results = 1, ValueRange operands = {}) {
    return keyedOp(--lastKey, results, operands);
  }

  Value constant(int64_t value) {
    return arith::ConstantIntOp::create(builder, builder.getUnknownLoc(), value,
                                        32);
  }

  /// Returns an uncacheable op's result that answers `value`. Unlike a
  /// constant, it costs a step in every query.
  Value uncached(int64_t value) {
    Operation *result = op();
    uncacheable(result);
    behaviors.at(result).evaluate = [value](Operation *, EvalScope &scope) {
      return SmallVector<Answer>{integer(scope.getAllocator(), value)};
    };
    return result->getResult(0);
  }

  int64_t lastKey = 0;
};

/// Caches `op` only when its `size` attribute is at least 100, as a model
/// may cache a large `linalg.matmul` but not a small one.
void sizeEligibility(Operation *op, int64_t size) {
  op->setAttr("size", Builder(op->getContext()).getI64IntegerAttr(size));
  behaviors.at(op).cacheable = [](Operation *op, EvalScope &) {
    auto size = op->getAttrOfType<IntegerAttr>("size");
    return size.getInt() >= 100;
  };
}

/// Caches `op` only when its first operand is known and at least 100.
void operandEligibility(Operation *op) {
  behaviors.at(op).cacheable = [](Operation *op,
                                  EvalScope &scope) -> EvalResult<bool> {
    Answer size = scope.query(op->getOperand(0));
    if (!size.completed())
      return size.status;
    const auto &value = size.getValue();
    return value && cast<IntEvalValue>(value).getValue().sge(100);
  };
}

/// Makes `op` answer with a new `OpaqueEvalValue` on every evaluation.
void opaque(Operation *op) {
  behaviors.at(op).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{OpaqueEvalValue::get(scope.getAllocator())};
  };
}

/// Answers `value` on every evaluation of `op`.
void answer(Operation *op, int64_t value) {
  behaviors.at(op).evaluate = [value](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), value)};
  };
}

/// Creates an `scf.execute_region` that yields the value `body` creates in it.
Value executeRegion(OpBuilder &builder, function_ref<Value()> body) {
  Location loc = builder.getUnknownLoc();
  auto op = scf::ExecuteRegionOp::create(builder, loc, builder.getI32Type());
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&op.getRegion().emplaceBlock());
  scf::YieldOp::create(builder, loc, body());
  return op.getResult(0);
}

void testSemanticReuseAndBudget() {
  Fixture f;
  Operation *first = f.keyedOp(1, 1, f.uncached(9));
  Operation *second = f.keyedOp(1, 1, f.uncached(9));

  // `first` costs 2 steps: its key evaluates its operand, which evaluating
  // `first` then reuses, and evaluating `first` takes the second.
  Engine oneStep(f.evalContext, 1);
  CHECK(oneStep.query(first->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(first).evaluations == 0);
  Engine twoSteps(f.evalContext, 2);
  CHECK(isInteger(twoSteps.query(first->getResult(0)), 7));
  CHECK(behaviors.at(first).evaluations == 1);

  // `second` costs only the step of its operand: comparing keys queries
  // nothing, so it hits.
  Engine noSteps(f.evalContext, 0);
  CHECK(noSteps.query(second->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(isInteger(oneStep.query(second->getResult(0)), 7));
  CHECK(behaviors.at(second).evaluations == 0);
  CHECK(behaviors.at(first).evaluations == 1);
}

void testPartialResultsAndLifetime() {
  Fixture f;
  Operation *op = f.op(3);
  behaviors.at(op).evaluate = [](Operation *op, EvalScope &scope) {
    if (behaviors.at(op).evaluations == 1)
      return SmallVector<Answer>{integer(scope.getAllocator(), 5),
                                 Answer::unknown(), EvalStatus::Exhausted};
    return SmallVector<Answer>{integer(scope.getAllocator(), 99),
                               integer(scope.getAllocator(), 99),
                               integer(scope.getAllocator(), 13)};
  };
  std::optional<Answer> first;
  {
    Engine engine(f.evalContext, 1);
    Answer exhausted = engine.query(op->getResult(2));
    CHECK(exhausted.status == EvalStatus::Exhausted);
    CHECK(!exhausted.isUnknown());
    first = engine.query(op->getResult(0));
    CHECK(isInteger(*first, 5));
    CHECK(engine.query(op->getResult(1)).isUnknown());
    CHECK(behaviors.at(op).evaluations == 1);
    // The signal left its slot empty, so evaluating again fills it, and keeps
    // the other answers.
    CHECK(isInteger(engine.query(op->getResult(2)), 13));
    CHECK(behaviors.at(op).evaluations == 2);
    CHECK(isInteger(engine.query(op->getResult(0)), 5));
    CHECK(engine.query(op->getResult(1)).isUnknown());
    CHECK(behaviors.at(op).evaluations == 2);
    CHECK(first->getValue().isCached());
    f.evalContext.clearCache();
    CHECK(isInteger(engine.query(op->getResult(0)), 99));
    CHECK(behaviors.at(op).evaluations == 3);
  }
  CHECK(isInteger(*first, 5));
}

void testPartialHitFills() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *candidate = f.keyedOp(1, 2);
  behaviors.at(candidate).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 7),
                               EvalStatus::NeedsOrder};
  };
  CHECK(engine.query(candidate->getResult(1)).status == EvalStatus::NeedsOrder);
  // `op` shares `candidate`'s entry, whose second slot is empty, so the lookup
  // misses and the evaluation fills the slot.
  Operation *op = f.keyedOp(1, 2);
  behaviors.at(op).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 7),
                               integer(scope.getAllocator(), 8)};
  };
  CacheOutcome outcome;
  CHECK(isInteger(engine.query(op->getResult(1), outcome), 8));
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(isInteger(engine.query(candidate->getResult(1), outcome), 8));
  CHECK(outcome == CacheOutcome::Hit);
  CHECK(behaviors.at(candidate).evaluations == 1);
}

void testSignalsAndOptOut() {
  for (EvalStatus signal :
       {EvalStatus::Exhausted, EvalStatus::NeedsOrder, EvalStatus::Cycle}) {
    Fixture f;
    Engine engine(f.evalContext, 1000);
    Operation *input = f.op();
    uncacheable(input);
    bool signals = true;
    behaviors.at(input).evaluate = [&](Operation *, EvalScope &scope) {
      if (signals)
        return SmallVector<Answer>{signal};
      return SmallVector<Answer>{integer(scope.getAllocator(), 1)};
    };
    Value later = f.uncached(2);
    Operation *op = f.op(1, {input->getResult(0), later});

    // The signal ends the key, so `later` isn't evaluated, and neither is
    // `op`.
    CacheOutcome outcome;
    Answer failure = engine.query(op->getResult(0), outcome);
    CHECK(failure.status == signal);
    CHECK(!failure.isUnknown());
    CHECK(outcome == CacheOutcome::None);
    CHECK(behaviors.at(later.getDefiningOp()).evaluations == 0);
    CHECK(behaviors.at(op).evaluations == 0);
    // Nothing was stored, so once the input completes, `op` misses.
    signals = false;
    CHECK(isInteger(engine.query(op->getResult(0), outcome), 7));
    CHECK(outcome == CacheOutcome::Miss);

    // Opting out skips the key, so an input `evaluate` doesn't query is never
    // evaluated.
    signals = true;
    unsigned inputEvaluations = behaviors.at(input).evaluations;
    Operation *optedOut = f.op(1, input->getResult(0));
    uncacheable(optedOut);
    answer(optedOut, 7);
    CHECK(isInteger(engine.query(optedOut->getResult(0)), 7));
    CHECK(isInteger(engine.query(optedOut->getResult(0)), 7));
    CHECK(behaviors.at(optedOut).evaluations == 2);
    CHECK(behaviors.at(input).evaluations == inputEvaluations);
  }
}

/// Queries enough new ops to rehash the cache's map.
void populate(Fixture &f, EvalScope &scope) {
  for (unsigned i = 0; i != 256; ++i)
    scope.query(f.op()->getResult(0));
}

void testInsertionDuringEvaluation() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  // Evaluating `op` rehashes the map, then stores `equal` under `op`'s key, so
  // `op`'s insertion finds that entry instead of adding another.
  Operation *op = f.keyedOp(1);
  Operation *equal = f.keyedOp(1);
  answer(equal, 8);
  behaviors.at(op).evaluate = [&](Operation *, EvalScope &scope) {
    populate(f, scope);
    scope.query(equal->getResult(0));
    return SmallVector<Answer>{integer(scope.getAllocator(), 7)};
  };
  CacheOutcome outcome;
  CHECK(isInteger(engine.query(op->getResult(0)), 8));
  CHECK(isInteger(engine.query(op->getResult(0), outcome), 8));
  CHECK(outcome == CacheOutcome::Hit);
  CHECK(behaviors.at(op).evaluations == 1);
  CHECK(behaviors.at(equal).evaluations == 1);

  // A partial entry found by the lookup is filled after a rehash, since its
  // slots don't move.
  Operation *partial = f.keyedOp(2, 2);
  behaviors.at(partial).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 7),
                               EvalStatus::NeedsOrder};
  };
  CHECK(engine.query(partial->getResult(0)).status == EvalStatus::Completed);
  Operation *filling = f.keyedOp(2, 2);
  behaviors.at(filling).evaluate = [&](Operation *, EvalScope &scope) {
    populate(f, scope);
    return SmallVector<Answer>{integer(scope.getAllocator(), 7),
                               integer(scope.getAllocator(), 9)};
  };
  CHECK(isInteger(engine.query(filling->getResult(1)), 9));
  CHECK(isInteger(engine.query(partial->getResult(1), outcome), 9));
  CHECK(outcome == CacheOutcome::Hit);
}

void testCycleThroughInput() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  // `back` is `op`'s input and queries `op`, so building `op`'s key re-enters
  // it.
  Operation *op = f.op();
  Operation *back = f.op();
  uncacheable(back);
  bool loops = true;
  behaviors.at(back).evaluate = [&](Operation *self, EvalScope &scope) {
    if (loops) {
      Answer answer = scope.query(op->getResult(0));
      if (!answer.completed())
        return answer.propagate(self);
    }
    return SmallVector<Answer>{integer(scope.getAllocator(), 3)};
  };
  op->setOperands(back->getResult(0));
  Answer cycle = engine.query(op->getResult(0));
  CHECK(cycle.status == EvalStatus::Cycle);
  CHECK(!cycle.isUnknown());
  CHECK(behaviors.at(op).evaluations == 0);
  // `op` must have left the active set on the early return.
  loops = false;
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
}

void testCycleThroughOperands() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  // The module body is a graph region, so a use-def cycle is valid IR.
  Operation *x = f.op();
  Operation *y = f.op(1, x->getResult(0));
  x->setOperands(y->getResult(0));
  // Each key evaluates the other op, so the cycle ends both lookups before
  // either op is evaluated.
  Answer cycle = engine.query(x->getResult(0));
  CHECK(cycle.status == EvalStatus::Cycle);
  CHECK(!cycle.isUnknown());
  CHECK(behaviors.at(x).evaluations == 0);
  CHECK(behaviors.at(y).evaluations == 0);
  x->setOperands({});
  CHECK(isInteger(engine.query(y->getResult(0)), 7));
}

void testSelfUseIsInner() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  // In a graph region, an op can use its own result. Its key numbers the use
  // as an inner value, so the lookup queries nothing and completes, and only
  // the evaluation re-enters `op`.
  Operation *op = f.op();
  op->setOperands(op->getResult(0));
  CacheOutcome outcome;
  CHECK(engine.query(op->getResult(0), outcome).status == EvalStatus::Cycle);
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(behaviors.at(op).evaluations == 1);
}

void testMemoizedOperand() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *n = f.op();
  uncacheable(n);
  Operation *a = f.op(1, n->getResult(0));
  // Building `a`'s key and evaluating `a` share one evaluation of `n`.
  CHECK(isInteger(engine.query(a->getResult(0)), 7));
  CHECK(behaviors.at(n).evaluations == 1);
  // The memo is per query, so the key evaluates `n` again, then hits.
  CHECK(isInteger(engine.query(a->getResult(0)), 7));
  CHECK(behaviors.at(n).evaluations == 2);
  CHECK(behaviors.at(a).evaluations == 1);
}

void testMemoizedSiblings() {
  Fixture f;
  Operation *pair = f.op(2);
  uncacheable(pair);
  Operation *sum = f.op(1, {pair->getResult(0), pair->getResult(1)});
  // One step for `pair` and one for `sum`: `pair`'s second result is recorded
  // when its first is evaluated.
  Engine twoSteps(f.evalContext, 2);
  CHECK(isInteger(twoSteps.query(sum->getResult(0)), 7));
  CHECK(behaviors.at(pair).evaluations == 1);
}

void testLazyOperands() {
  Fixture f;
  Engine oneStep(f.evalContext, 1);
  Operation *operand = f.op();
  // A key evaluates every operand, so an op that may not need one opts out.
  Operation *ignoring = f.op(1, operand->getResult(0));
  uncacheable(ignoring);
  answer(ignoring, 8);
  // `ignoring` never queries its operand, so it costs only its own step.
  CHECK(isInteger(oneStep.query(ignoring->getResult(0)), 8));
  CHECK(behaviors.at(operand).evaluations == 0);
  // The step is charged before `evaluate` runs, so `user` runs and its operand
  // exhausts.
  Operation *user = f.op(1, operand->getResult(0));
  uncacheable(user);
  CHECK(oneStep.query(user->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(user).evaluations == 1);
  CHECK(behaviors.at(operand).evaluations == 0);
  // A cached op's key evaluates the operand first, which takes the step.
  Operation *keyed = f.op(1, operand->getResult(0));
  CHECK(oneStep.query(keyed->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(keyed).evaluations == 0);
  CHECK(behaviors.at(operand).evaluations == 1);
}

void testEligibilityFromAttributes() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *large = f.op();
  Operation *small = f.op();
  sizeEligibility(large, 1000);
  sizeEligibility(small, 10);
  CacheOutcome outcome;
  CHECK(isInteger(engine.query(large->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(isInteger(engine.query(large->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Hit);
  CHECK(behaviors.at(large).evaluations == 1);
  // `small` is neither looked up nor inserted, so every query evaluates it.
  for (unsigned i = 0; i != 2; ++i) {
    Answer answer = engine.query(small->getResult(0), outcome);
    CHECK(isInteger(answer, 7));
    CHECK(!answer.getValue().isCached());
    CHECK(outcome == CacheOutcome::None);
  }
  CHECK(behaviors.at(small).evaluations == 2);
}

void testEligibilityFromOperands() {
  Fixture f;
  Operation *large = f.op(1, f.uncached(1000));
  Operation *small = f.op(1, f.uncached(10));
  operandEligibility(large);
  operandEligibility(small);

  // A miss on `large` costs 2 steps: eligibility evaluates its operand, which
  // evaluation then reuses, and evaluating `large` takes the second.
  Engine noSteps(f.evalContext, 0);
  CHECK(noSteps.query(large->getResult(0)).status == EvalStatus::Exhausted);
  Engine oneStep(f.evalContext, 1);
  CHECK(oneStep.query(large->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(large).evaluations == 0);
  Engine twoSteps(f.evalContext, 2);
  CHECK(isInteger(twoSteps.query(large->getResult(0)), 7));
  // A hit costs only the operand.
  CHECK(isInteger(oneStep.query(large->getResult(0)), 7));
  CHECK(behaviors.at(large).evaluations == 1);

  CHECK(isInteger(twoSteps.query(small->getResult(0)), 7));
  CHECK(isInteger(twoSteps.query(small->getResult(0)), 7));
  CHECK(behaviors.at(small).evaluations == 2);
}

void testEligibilitySharesMemo() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *n = f.op();
  uncacheable(n);
  behaviors.at(n).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 1000)};
  };
  Operation *a = f.op(1, n->getResult(0));
  operandEligibility(a);
  // Eligibility, the key and evaluating `a` share one evaluation of `n`.
  CHECK(isInteger(engine.query(a->getResult(0)), 7));
  CHECK(behaviors.at(n).evaluations == 1);
}

void testEligibilitySignals() {
  for (EvalStatus signal :
       {EvalStatus::Exhausted, EvalStatus::NeedsOrder, EvalStatus::Cycle}) {
    Fixture f;
    Engine engine(f.evalContext, 1000);
    Operation *op = f.op();
    behaviors.at(op).cacheable = [signal](Operation *,
                                          EvalScope &) -> EvalResult<bool> {
      return signal;
    };
    CacheOutcome outcome;
    Answer answer = engine.query(op->getResult(0), outcome);
    CHECK(answer.status == signal);
    CHECK(!answer.isUnknown());
    CHECK(outcome == CacheOutcome::None);
    CHECK(behaviors.at(op).evaluations == 0);
  }
}

void testCycleThroughEligibility() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *op = f.op();
  behaviors.at(op).cacheable = [](Operation *op,
                                  EvalScope &scope) -> EvalResult<bool> {
    Answer answer = scope.query(op->getResult(0));
    if (!answer.completed())
      return answer.status;
    return true;
  };
  Answer cycle = engine.query(op->getResult(0));
  CHECK(cycle.status == EvalStatus::Cycle);
  CHECK(behaviors.at(op).eligibilityChecks == 1);
  CHECK(behaviors.at(op).evaluations == 0);
  // `op` must have left the active set on the early return.
  behaviors.at(op).cacheable = Behavior().cacheable;
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
}

void testUncacheableOpsCanBeKeyed() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  SmallVector<Operation *> nested;
  // Each region op holds an uncacheable op whose key is `key`.
  auto withNested = [&](int64_t key) {
    return executeRegion(f.builder, [&] {
      Operation *op = f.keyedOp(key, 1, f.constant(9));
      uncacheable(op);
      nested.push_back(op);
      return op->getResult(0);
    });
  };
  Value first = withNested(1);
  Value second = withNested(1);
  Value other = withNested(2);
  CacheOutcome outcome;
  for (auto [value, expected] : {std::pair(first, CacheOutcome::Miss),
                                 std::pair(second, CacheOutcome::Hit),
                                 std::pair(other, CacheOutcome::Miss)}) {
    CHECK(isInteger(engine.query(value, outcome), 7));
    CHECK(outcome == expected);
  }
  // Nested ops are only structure in their region op's key.
  for (Operation *op : nested) {
    CHECK(behaviors.at(op).eligibilityChecks == 0);
    CHECK(behaviors.at(op).evaluations == 0);
  }
}

void testIdentityValueKeys() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *producer = f.keyedOp(1);
  Operation *equal = f.keyedOp(1);
  opaque(producer);
  opaque(equal);
  Operation *a = f.keyedOp(2, 1, producer->getResult(0));
  Operation *b = f.keyedOp(2, 1, equal->getResult(0));
  // `producer`'s answer is the instance the cache retained, so `a`'s key is
  // the same in every query.
  CacheOutcome outcome;
  CHECK(isInteger(engine.query(a->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(isInteger(engine.query(a->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Hit);
  // `equal` hits `producer`'s entry, so `b`'s key holds the same instance as
  // `a`'s and `b` hits too.
  CHECK(isInteger(engine.query(b->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Hit);
  CHECK(behaviors.at(a).evaluations == 1);
  CHECK(behaviors.at(b).evaluations == 0);
  CHECK(behaviors.at(producer).evaluations == 1);
  CHECK(behaviors.at(equal).evaluations == 0);

  // Uncached producers answer with a new instance each, so keys holding equal
  // contents still differ.
  Operation *first = f.op();
  Operation *second = f.op();
  for (Operation *op : {first, second}) {
    opaque(op);
    uncacheable(op);
  }
  Operation *c = f.keyedOp(3, 1, first->getResult(0));
  Operation *d = f.keyedOp(3, 1, second->getResult(0));
  CHECK(isInteger(engine.query(c->getResult(0)), 7));
  CHECK(isInteger(engine.query(d->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(behaviors.at(d).evaluations == 1);
}

void testSideEffectsOptOut() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Location loc = f.builder.getUnknownLoc();
  Value constant = f.constant(9);
  Value memory = memref::AllocaOp::create(
      f.builder, loc, MemRefType::get({}, f.builder.getI32Type()));
  using Make = std::function<Value()>;
  Make add = [&]() -> Value {
    return arith::AddIOp::create(f.builder, loc, constant, constant);
  };
  Make load = [&]() -> Value {
    return memref::LoadOp::create(f.builder, loc, memory, ValueRange());
  };
  auto inRegion = [&](Make body) -> Make {
    return [&, body] { return executeRegion(f.builder, body); };
  };

  // Pure ops share an entry, and so do region ops whose bodies are pure.
  CacheOutcome outcome;
  for (const Make &make : {add, inRegion(add)}) {
    Value first = make();
    Value second = make();
    CHECK(isInteger(engine.query(first, outcome), 7));
    CHECK(outcome == CacheOutcome::Miss);
    CHECK(isInteger(engine.query(second, outcome), 7));
    CHECK(outcome == CacheOutcome::Hit);
  }

  // A load can read different memory than an equal load, so loads, and region
  // ops that load, are evaluated on every query.
  for (const Make &make : {load, inRegion(load)}) {
    Value first = make();
    Value second = make();
    for (Value value : {first, second, first}) {
      CHECK(isInteger(engine.query(value, outcome), 7));
      CHECK(outcome == CacheOutcome::None);
    }
    CHECK(behaviors.at(first.getDefiningOp()).evaluations == 2);
    CHECK(behaviors.at(second.getDefiningOp()).evaluations == 1);
  }
}

void testConstants() {
  Fixture f;
  Location loc = f.builder.getUnknownLoc();
  auto poisonOf = [&](Type type) -> Value {
    return ub::PoisonOp::create(f.builder, loc, type);
  };
  Value i32 = poisonOf(f.builder.getI32Type());
  Value sameI32 = poisonOf(f.builder.getI32Type());
  Value f32 = poisonOf(f.builder.getF32Type());
  // Constants cost no step.
  Engine noSteps(f.evalContext, 0);
  CacheOutcome outcome;

  Answer first = noSteps.query(i32, outcome);
  CHECK(first.completed() && first.getValue().isCached());
  CHECK(outcome == CacheOutcome::Miss);
  // Another op with the same attribute and type shares the value.
  Answer shared = noSteps.query(sameI32, outcome);
  CHECK(outcome == CacheOutcome::Hit);
  CHECK(shared.completed() && shared.getValue() == first.getValue());
  // The same attribute with another type gets its own value.
  Answer other = noSteps.query(f32, outcome);
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(other.completed() && other.getValue() != first.getValue());
  CHECK(poison.conversions == 2);

  f.evalContext.clearCache();
  CHECK(noSteps.query(i32, outcome).completed());
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(poison.conversions == 3);

  // Unknowns are stored too, so they're converted once.
  Value unknown = poisonOf(f.builder.getI64Type());
  poison.known = false;
  for (CacheOutcome expected : {CacheOutcome::Miss, CacheOutcome::Hit}) {
    CHECK(noSteps.query(unknown, outcome).isUnknown());
    CHECK(outcome == expected);
  }
  CHECK(poison.conversions == 4);

  // An attribute that doesn't opt in skips the map, though `i32` is in it.
  poison.known = true;
  poison.cacheable = false;
  for (unsigned i = 0; i != 2; ++i) {
    Answer answer = noSteps.query(i32, outcome);
    CHECK(answer.completed() && !answer.getValue().isCached());
    CHECK(outcome == CacheOutcome::None);
  }
  CHECK(poison.conversions == 6);
}

/// Makes `op` answer its operand plus 1, or unknown when the operand is.
void addOne(Operation *op) {
  behaviors.at(op).evaluate = [](Operation *op, EvalScope &scope) {
    Answer operand = scope.query(op->getOperand(0));
    if (!operand.completed())
      return operand.propagate(op);
    if (operand.isUnknown())
      return SmallVector<Answer>{operand};
    const APInt &value = cast<IntEvalValue>(operand.getValue()).getValue();
    return SmallVector<Answer>{
        IntEvalValue::get(scope.getAllocator(), value + 1)};
  };
}

void testLoopBodyReadsCache() {
  Fixture f;
  Location loc = f.builder.getUnknownLoc();
  Operation *top = f.keyedOp(1, 1, f.constant(1));
  Value zero = f.constant(0);
  auto loop = scf::ForOp::create(f.builder, loc, zero, f.constant(3),
                                 f.constant(1), zero);
  Operation *body;
  {
    OpBuilder::InsertionGuard guard(f.builder);
    f.builder.setInsertionPointToStart(loop.getBody());
    body = f.keyedOp(1, 1, loop.getRegionIterArgs()[0]);
    scf::YieldOp::create(f.builder, loc, body->getResult(0));
  }
  addOne(top);
  addOne(body);
  Engine engine(f.evalContext, 1000);
  CacheOutcome outcome;
  CHECK(isInteger(engine.query(top->getResult(0), outcome), 2));
  CHECK(outcome == CacheOutcome::Miss);

  // Each iteration looks the body up under its own bindings. With the
  // iteration argument at 1, its key matches `top`, so it hits.
  Behavior &behavior = behaviors.at(body);
  CHECK(isInteger(engine.query(loop.getResult(0), outcome), 3));
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(behavior.eligibilityChecks == 3);
  CHECK(behavior.evaluations == 2);

  // The loop itself is cached.
  CHECK(isInteger(engine.query(loop.getResult(0), outcome), 3));
  CHECK(outcome == CacheOutcome::Hit);
  CHECK(behavior.evaluations == 2);

  // Outside an iteration, the iteration argument is unknown, so is the body.
  // The unknown is stored, so a second query hits it.
  for (CacheOutcome expected : {CacheOutcome::Miss, CacheOutcome::Hit}) {
    CHECK(engine.query(body->getResult(0), outcome).isUnknown());
    CHECK(outcome == expected);
  }
  CHECK(behavior.evaluations == 3);
}

void testUnknownUnderUnboundArgument() {
  Fixture f;
  Location loc = f.builder.getUnknownLoc();
  Value zero = f.constant(0);
  auto loop = scf::ForOp::create(f.builder, loc, zero, f.constant(3),
                                 f.constant(1), zero);
  Operation *body;
  {
    OpBuilder::InsertionGuard guard(f.builder);
    f.builder.setInsertionPointToStart(loop.getBody());
    body = f.op(1, loop.getRegionIterArgs()[0]);
    scf::YieldOp::create(f.builder, loc, body->getResult(0));
  }
  addOne(body);
  Engine engine(f.evalContext, 1000);
  CacheOutcome outcome;
  // With the iteration argument unbound, the body is unknown, and its entry is
  // keyed by the unknown argument.
  CHECK(engine.query(body->getResult(0), outcome).isUnknown());
  CHECK(outcome == CacheOutcome::Miss);
  // Each iteration binds the argument, so the body misses that entry.
  CHECK(isInteger(engine.query(loop.getResult(0)), 3));
  CHECK(behaviors.at(body).evaluations == 4);
}

void testLoopFreezesCache() {
  Fixture f;
  Location loc = f.builder.getUnknownLoc();
  Value memory = memref::AllocaOp::create(
      f.builder, loc, MemRefType::get({}, f.builder.getI32Type()));
  Value zero = f.constant(0);
  auto loop = scf::ForOp::create(f.builder, loc, zero, f.constant(3),
                                 f.constant(1), zero);
  Operation *body;
  {
    OpBuilder::InsertionGuard guard(f.builder);
    f.builder.setInsertionPointToStart(loop.getBody());
    // The load makes the loop uncacheable, so every query runs it.
    memref::LoadOp::create(f.builder, loc, memory, ValueRange());
    body = f.op(1, loop.getRegionIterArgs()[0]);
    scf::YieldOp::create(f.builder, loc, body->getResult(0));
  }
  addOne(body);

  Engine engine(f.evalContext, 1000);
  CacheOutcome outcome;
  Behavior &behavior = behaviors.at(body);
  CHECK(isInteger(engine.query(loop.getResult(0), outcome), 3));
  CHECK(outcome == CacheOutcome::None);
  CHECK(behavior.evaluations == 3);

  // The first run stored nothing, so the second finds no entry for the body.
  CHECK(isInteger(engine.query(loop.getResult(0), outcome), 3));
  CHECK(behavior.evaluations == 6);
}

void testCapturedBinding() {
  Fixture f;
  Location loc = f.builder.getUnknownLoc();
  // The loop never runs: `probe` binds its body without a freeze.
  Value zero = f.constant(0);
  auto loop = scf::ForOp::create(f.builder, loc, zero, zero, zero);
  Value captures;
  {
    OpBuilder::InsertionGuard guard(f.builder);
    f.builder.setInsertionPointToStart(loop.getBody());
    captures = executeRegion(f.builder, [&] {
      return f.op(1, loop.getInductionVar())->getResult(0);
    });
  }
  Operation *region = captures.getDefiningOp();
  int64_t bound = 0;
  Operation *probe = f.op();
  uncacheable(probe);
  behaviors.at(probe).evaluate = [&](Operation *, EvalScope &scope) {
    BlockBinding binding =
        scope.bind(loop.getBody(), {integer(scope.getAllocator(), bound)});
    return SmallVector<Answer>{scope.query(captures)};
  };

  // `region` has no operands, but its key holds the bound value its body
  // captures, so an entry stored under 5 is only hit under 5.
  Engine engine(f.evalContext, 1000);
  for (auto [value, evaluations] :
       {std::pair(5, 1), std::pair(5, 1), std::pair(6, 2), std::pair(5, 2)}) {
    bound = value;
    CHECK(isInteger(engine.query(probe->getResult(0)), 7));
    CHECK(behaviors[region].evaluations == unsigned(evaluations));
  }
}

void testCapturedValues() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Value c2 = f.constant(2);
  Value c3 = f.constant(3);
  Value two = f.constant(2);
  // The region ops have no operands, and their bodies capture `value`.
  auto capturing = [&](Value value) {
    return executeRegion(f.builder,
                         [&] { return f.keyedOp(1, 1, value)->getResult(0); });
  };
  CacheOutcome outcome;
  for (auto [value, expected] :
       {std::pair(capturing(c2), CacheOutcome::Miss),
        std::pair(capturing(c3), CacheOutcome::Miss),
        std::pair(capturing(two), CacheOutcome::Hit)}) {
    CHECK(isInteger(engine.query(value, outcome), 7));
    CHECK(outcome == expected);
  }
}

void testCopiedPropertiesAreFreed() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  // The label is too long for `std::string`'s inline buffer, so the cache's
  // copy of the nested op's properties owns heap memory, which only
  // `destroyOpProperties` frees. Under LSan, skipping it fails this test.
  auto labeled = [&] {
    return executeRegion(f.builder, [&]() -> Value {
      return interp_test::PropOp::create(
          f.builder, f.builder.getUnknownLoc(), f.builder.getI32Type(),
          "a label longer than any inline buffer");
    });
  };
  Value first = labeled();
  Value second = labeled();
  // Properties are compared by content, so the second op hits.
  CacheOutcome outcome;
  for (auto [value, expected] : {std::pair(first, CacheOutcome::Miss),
                                 std::pair(second, CacheOutcome::Hit)}) {
    CHECK(isInteger(engine.query(value, outcome), 7));
    CHECK(outcome == expected);
  }
}

void testErasedOp() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Location loc = f.builder.getUnknownLoc();
  Value c2 = f.constant(2);
  // The overflow flags are properties, which the entry copies.
  auto add = [&]() -> Operation * {
    return arith::AddIOp::create(f.builder, loc, c2, c2,
                                 arith::IntegerOverflowFlags::nsw);
  };
  Operation *erased = add();
  CHECK(isInteger(engine.query(erased->getResult(0)), 7));
  behaviors.erase(erased);
  erased->erase();
  // The entry holds no reference to the erased op, so an equal op hits it.
  Operation *equal = add();
  CacheOutcome outcome;
  CHECK(isInteger(engine.query(equal->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Hit);
  CHECK(behaviors[equal].evaluations == 0);
}

} // namespace

int main() {
  testSemanticReuseAndBudget();
  testPartialResultsAndLifetime();
  testPartialHitFills();
  testSignalsAndOptOut();
  testInsertionDuringEvaluation();
  testCycleThroughInput();
  testCycleThroughOperands();
  testSelfUseIsInner();
  testMemoizedOperand();
  testMemoizedSiblings();
  testLazyOperands();
  testEligibilityFromAttributes();
  testEligibilityFromOperands();
  testEligibilitySharesMemo();
  testEligibilitySignals();
  testCycleThroughEligibility();
  testUncacheableOpsCanBeKeyed();
  testIdentityValueKeys();
  testSideEffectsOptOut();
  testConstants();
  testLoopBodyReadsCache();
  testUnknownUnderUnboundArgument();
  testLoopFreezesCache();
  testCapturedBinding();
  testCapturedValues();
  testCopiedPropertiesAreFreed();
  testErasedOp();
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
