#include "interpreter/Dialects/Builtin/BuiltinAttrEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
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
         cast<IntEvalValue>(*answer.getValue()).getValue() == value;
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
  unsigned hashes = 0;
  unsigned evaluations = 0;
  std::function<EvalResult<bool>(Operation *, EvalScope &)> cacheable =
      [](Operation *, EvalScope &) { return true; };
  std::function<EvalResult<llvm::hash_code>(Operation *, EvalScope &)> hash =
      [](Operation *op, EvalScope &) { return llvm::hash_value(op); };
  std::function<EvalResult<bool>(Operation *, Operation *, EvalScope &)> equal =
      [](Operation *op, Operation *other, EvalScope &) { return op == other; };
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

struct TestModel
    : EvaluableOpInterface::ExternalModel<TestModel,
                                          UnrealizedConversionCastOp> {
  EvalResult<llvm::hash_code> getHash(Operation *op, EvalScope &scope) const {
    auto &behavior = behaviors.at(op);
    ++behavior.hashes;
    return behavior.hash(op, scope);
  }

  EvalResult<bool> isEquivalentTo(Operation *op, Operation *other,
                                  EvalScope &scope) const {
    return behaviors.at(op).equal(op, other, scope);
  }

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
  return result;
}

struct Fixture {
  MLIRContext context{registry()};
  EvalContext evalContext{&context};
  OwningOpRef<ModuleOp> module{ModuleOp::create(UnknownLoc::get(&context))};
  OpBuilder builder{module->getBody(), module->getBody()->begin()};

  Fixture() {
    context.loadDialect<arith::ArithDialect, memref::MemRefDialect,
                        scf::SCFDialect, ub::UBDialect>();
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

  Operation *op(unsigned results = 1, ValueRange operands = {}) {
    OperationState state(builder.getUnknownLoc(),
                         UnrealizedConversionCastOp::getOperationName());
    state.addTypes(SmallVector<Type>(results, builder.getI32Type()));
    state.addOperands(operands);
    Operation *result = builder.create(state);
    behaviors.try_emplace(result);
    return result;
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

  Operation *hashedOp(llvm::hash_code key, unsigned results = 1) {
    Operation *result = op(results);
    behaviors.at(result).hash = [key](Operation *, EvalScope &) { return key; };
    return result;
  }
};

void semanticKey(Operation *op) {
  behaviors.at(op).hash = [](Operation *op,
                             EvalScope &scope) -> EvalResult<llvm::hash_code> {
    Answer answer = scope.query(op->getOperand(0));
    if (!answer.completed())
      return answer.status;
    if (const auto &value = answer.getValue())
      return value->hash();
    return llvm::hash_value(op->getOperand(0).getAsOpaquePointer());
  };
  behaviors.at(op).equal = [](Operation *op, Operation *other,
                              EvalScope &scope) -> EvalResult<bool> {
    if (op->getName() != other->getName() || other->getNumOperands() != 1 ||
        op->getResultTypes() != other->getResultTypes())
      return false;
    Answer lhs = scope.query(op->getOperand(0));
    if (!lhs.completed())
      return lhs.status;
    Answer rhs = scope.query(other->getOperand(0));
    if (!rhs.completed())
      return rhs.status;
    if (lhs.isUnknown() || rhs.isUnknown())
      return lhs.isUnknown() && rhs.isUnknown() &&
             op->getOperand(0) == other->getOperand(0);
    return lhs.getValue()->isEqual(*rhs.getValue());
  };
}

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
    return value && cast<IntEvalValue>(*value).getValue().sge(100);
  };
}

/// Makes `op` answer with a new `OpaqueEvalValue` on every evaluation.
void opaque(Operation *op) {
  behaviors.at(op).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{OpaqueEvalValue::get(scope.getAllocator())};
  };
}

void testSemanticReuseAndBudget() {
  Fixture f;
  Operation *first = f.op(1, f.uncached(9));
  Operation *second = f.op(1, f.uncached(9));
  semanticKey(first);
  semanticKey(second);

  // `first` costs 2 steps: hashing evaluates its operand, which evaluating
  // `first` then reuses, and evaluating `first` takes the second.
  Engine oneStep(f.evalContext, 1);
  CHECK(oneStep.query(first->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(first).evaluations == 0);
  Engine twoSteps(f.evalContext, 2);
  CHECK(isInteger(twoSteps.query(first->getResult(0)), 7));
  CHECK(behaviors.at(first).evaluations == 1);

  // `second` also costs 2 steps: hashing evaluates its operand, and comparing
  // with `first` evaluates `first`'s operand, then hits.
  CHECK(oneStep.query(second->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(isInteger(twoSteps.query(second->getResult(0)), 7));
  CHECK(behaviors.at(second).evaluations == 0);
  CHECK(behaviors.at(first).evaluations == 1);
}

void testZeroHashCollision() {
  Fixture f;
  Engine engine(f.evalContext, 1);
  EvalScope scope(engine);
  Operation *first = f.hashedOp(llvm::hash_code(0));
  Operation *second = f.hashedOp(llvm::hash_code(0));
  behaviors.at(second).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 8)};
  };
  auto comparison =
      cast<EvaluableOpInterface>(second).isEquivalentTo(first, scope);
  CHECK(comparison.completed());
  CHECK(!comparison.getValue());
  CHECK(isInteger(engine.query(first->getResult(0)), 7));
  CHECK(isInteger(engine.query(second->getResult(0)), 8));
  CHECK(behaviors.at(first).evaluations == 1);
  CHECK(behaviors.at(second).evaluations == 1);
  Engine noSteps(f.evalContext, 0);
  CHECK(isInteger(noSteps.query(first->getResult(0)), 7));
  CHECK(isInteger(noSteps.query(second->getResult(0)), 8));
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
    CHECK(isInteger(engine.query(op->getResult(2)), 13));
    CHECK(behaviors.at(op).evaluations == 2);
    CHECK(isInteger(engine.query(op->getResult(0)), 5));
    CHECK(engine.query(op->getResult(1)).isUnknown());
    CHECK(first->getValue()->isCached());
    f.evalContext.clearCache();
    CHECK(isInteger(engine.query(op->getResult(0)), 99));
    CHECK(behaviors.at(op).evaluations == 3);
  }
  CHECK(isInteger(*first, 5));
}

void testPartialHitFillsWithoutSearching() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key, 2);
  behaviors.at(candidate).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 7),
                               EvalStatus::NeedsOrder};
  };
  CHECK(engine.query(candidate->getResult(1)).status == EvalStatus::NeedsOrder);
  Operation *op = f.hashedOp(key, 2);
  unsigned comparisons = 0;
  behaviors.at(op).equal = [&](Operation *, Operation *other, EvalScope &) {
    ++comparisons;
    return other == candidate;
  };
  behaviors.at(op).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 7),
                               integer(scope.getAllocator(), 8)};
  };
  CHECK(isInteger(engine.query(op->getResult(1)), 8));
  CHECK(behaviors.at(op).hashes == 1);
  CHECK(comparisons == 1);
  CHECK(isInteger(engine.query(candidate->getResult(1)), 8));
  CHECK(behaviors.at(candidate).evaluations == 1);
}

void testMissHashesOnce() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *op = f.op(3);
  Answer answer = engine.query(op->getResult(0));
  CHECK(isInteger(answer, 7));
  CHECK(answer.getValue()->isCached());
  CHECK(behaviors.at(op).hashes == 1);
}

void testSignalsAndOptOut() {
  for (EvalStatus signal :
       {EvalStatus::Exhausted, EvalStatus::NeedsOrder, EvalStatus::Cycle}) {
    Fixture f;
    Engine engine(f.evalContext, 1000);
    Operation *unhashable = f.op();
    behaviors.at(unhashable).hash =
        [signal](Operation *, EvalScope &) -> EvalResult<llvm::hash_code> {
      return signal;
    };
    Answer hashFailure = engine.query(unhashable->getResult(0));
    CHECK(hashFailure.status == signal);
    CHECK(!hashFailure.isUnknown());
    CHECK(behaviors.at(unhashable).evaluations == 0);

    // An equivalence signal skips the candidate instead of failing the lookup.
    auto key = llvm::hash_code(123);
    Operation *candidate = f.hashedOp(key);
    CHECK(isInteger(engine.query(candidate->getResult(0)), 7));
    Operation *op = f.hashedOp(key);
    behaviors.at(op).equal = [signal](Operation *, Operation *,
                                      EvalScope &) -> EvalResult<bool> {
      return signal;
    };
    Answer skipped = engine.query(op->getResult(0));
    CHECK(isInteger(skipped, 7));
    CHECK(skipped.getValue()->isCached());
    CHECK(behaviors.at(op).evaluations == 1);
    CHECK(isInteger(engine.query(op->getResult(0)), 7));
    CHECK(behaviors.at(op).evaluations == 1);

    // Opting out skips hashing, so the signaling hash is never called.
    uncacheable(unhashable);
    CHECK(isInteger(engine.query(unhashable->getResult(0)), 7));
    CHECK(isInteger(engine.query(unhashable->getResult(0)), 7));
    CHECK(behaviors.at(unhashable).evaluations == 2);
    CHECK(behaviors.at(unhashable).hashes == 1);
  }
}

/// Queries enough new operations to rehash the map and grow `key`'s bucket,
/// then queries `equivalent`.
void populate(Fixture &f, EvalScope &scope, llvm::hash_code key,
              Operation *equivalent) {
  for (unsigned i = 0; i != 256; ++i)
    scope.query(f.hashedOp(llvm::hash_code(1000 + i))->getResult(0));
  for (unsigned i = 0; i != 32; ++i)
    scope.query(f.hashedOp(key)->getResult(0));
  scope.query(equivalent->getResult(0));
}

Operation *equivalentTo(Fixture &f, llvm::hash_code key, Operation *op) {
  Operation *equivalent = f.hashedOp(key);
  behaviors.at(equivalent).equal = [op](Operation *, Operation *other,
                                        EvalScope &) { return other == op; };
  behaviors.at(equivalent).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 8)};
  };
  return equivalent;
}

void testNestedInsertionDuringLookup() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key);
  CHECK(isInteger(engine.query(candidate->getResult(0)), 7));
  Operation *query = f.hashedOp(key);
  Operation *equivalent = equivalentTo(f, key, query);
  bool populated = false;
  behaviors.at(query).equal = [&](Operation *, Operation *other,
                                  EvalScope &scope) {
    if (!populated) {
      populated = true;
      populate(f, scope, key, equivalent);
    }
    return other == equivalent;
  };
  CHECK(isInteger(engine.query(query->getResult(0)), 8));
  CHECK(populated);
  CHECK(behaviors.at(query).evaluations == 0);
  CHECK(behaviors.at(equivalent).evaluations == 1);
}

void testNestedInsertionDuringInsert() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  auto key = llvm::hash_code(123);
  Operation *op = f.hashedOp(key);
  Operation *x = f.hashedOp(key);
  Operation *equivalent = equivalentTo(f, key, op);
  behaviors.at(op).evaluate = [x](Operation *, EvalScope &scope) {
    scope.query(x->getResult(0));
    return SmallVector<Answer>{integer(scope.getAllocator(), 7)};
  };
  bool populated = false;
  behaviors.at(op).equal = [&](Operation *, Operation *other,
                               EvalScope &scope) {
    if (other == x && !populated) {
      populated = true;
      populate(f, scope, key, equivalent);
    }
    return other == equivalent;
  };
  CHECK(isInteger(engine.query(op->getResult(0)), 8));
  CHECK(populated);
  CHECK(behaviors.at(op).evaluations == 1);
  CHECK(behaviors.at(equivalent).evaluations == 1);
}

void testInsertResumesSearch() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key);
  CHECK(isInteger(engine.query(candidate->getResult(0)), 7));
  Operation *op = f.hashedOp(key);
  Operation *x = f.hashedOp(key);
  behaviors.at(op).evaluate = [x](Operation *, EvalScope &scope) {
    scope.query(x->getResult(0));
    return SmallVector<Answer>{integer(scope.getAllocator(), 7)};
  };
  std::map<Operation *, unsigned> comparisons;
  behaviors.at(op).equal = [&](Operation *, Operation *other, EvalScope &) {
    ++comparisons[other];
    return false;
  };
  Answer answer = engine.query(op->getResult(0));
  CHECK(isInteger(answer, 7));
  CHECK(answer.getValue()->isCached());
  CHECK(comparisons[candidate] == 1);
  CHECK(comparisons[x] == 1);
}

void testEquivalenceConsumesBudget() {
  Fixture f;
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key);
  Engine large(f.evalContext, 1000);
  CHECK(isInteger(large.query(candidate->getResult(0)), 7));
  Operation *n1 = f.op();
  Operation *n2 = f.op();
  uncacheable(n1);
  uncacheable(n2);
  Operation *op = f.hashedOp(key);
  behaviors.at(op).equal = [n1, n2](Operation *, Operation *,
                                    EvalScope &scope) {
    scope.query(n1->getResult(0));
    scope.query(n2->getResult(0));
    return false;
  };
  // Comparing with `candidate` spends both steps, and skipping it refunds
  // nothing.
  Engine twoSteps(f.evalContext, 2);
  CHECK(twoSteps.query(op->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(op).evaluations == 0);
  Engine threeSteps(f.evalContext, 3);
  CHECK(isInteger(threeSteps.query(op->getResult(0)), 7));
}

void testResultCountFilter() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key, 2);
  CHECK(isInteger(engine.query(candidate->getResult(0)), 7));
  Operation *op = f.hashedOp(key);
  unsigned comparisons = 0;
  // Deliberately wrong: claims equivalence with every candidate.
  behaviors.at(op).equal = [&](Operation *, Operation *, EvalScope &) {
    ++comparisons;
    return true;
  };
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
  CHECK(comparisons == 0);
  CHECK(behaviors.at(op).evaluations == 1);
}

void testInsertionAfterEvaluation() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  auto key = llvm::hash_code(123);
  Operation *op = f.hashedOp(key);
  Operation *equivalent = equivalentTo(f, key, op);
  behaviors.at(op).equal = [equivalent](Operation *, Operation *other,
                                        EvalScope &) {
    return other == equivalent;
  };
  behaviors.at(op).evaluate = [equivalent](Operation *, EvalScope &scope) {
    scope.query(equivalent->getResult(0));
    return SmallVector<Answer>{integer(scope.getAllocator(), 7)};
  };
  CHECK(isInteger(engine.query(op->getResult(0)), 8));
  CHECK(behaviors.at(op).hashes == 1);
  CHECK(isInteger(engine.query(op->getResult(0)), 8));
  CHECK(behaviors.at(op).evaluations == 1);
}

void testCycleThroughEquivalence() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key);
  CHECK(isInteger(engine.query(candidate->getResult(0)), 7));
  Operation *op = f.hashedOp(key);
  std::optional<EvalStatus> recorded;
  behaviors.at(op).equal = [&](Operation *op, Operation *,
                               EvalScope &scope) -> EvalResult<bool> {
    Answer answer = scope.query(op->getResult(0));
    recorded = answer.status;
    if (!answer.completed())
      return answer.status;
    return false;
  };
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
  CHECK(recorded == EvalStatus::Cycle);
  CHECK(behaviors.at(op).evaluations == 1);
}

void testCycleThroughHash() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *op = f.op();
  behaviors.at(op).hash = [](Operation *op,
                             EvalScope &scope) -> EvalResult<llvm::hash_code> {
    Answer answer = scope.query(op->getResult(0));
    if (!answer.completed())
      return answer.status;
    return llvm::hash_value(op);
  };
  Answer cycle = engine.query(op->getResult(0));
  CHECK(cycle.status == EvalStatus::Cycle);
  CHECK(!cycle.isUnknown());
  CHECK(behaviors.at(op).evaluations == 0);
  // `op` must have left the active set on the early return.
  behaviors.at(op).hash = Behavior().hash;
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
}

void testCycleThroughOperands() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  // The module body is a graph region, so a use-def cycle is valid IR.
  Operation *x = f.op();
  Operation *y = f.op(1, x->getResult(0));
  x->setOperands(y->getResult(0));
  Answer cycle = engine.query(x->getResult(0));
  CHECK(cycle.status == EvalStatus::Cycle);
  CHECK(!cycle.isUnknown());
  CHECK(behaviors.at(x).evaluations == 1);
  CHECK(behaviors.at(y).evaluations == 1);
  x->setOperands({});
  CHECK(isInteger(engine.query(y->getResult(0)), 7));
}

void testMemoizedOperand() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *n = f.op();
  uncacheable(n);
  Operation *a = f.op(1, n->getResult(0));
  semanticKey(a);
  // Hashing and evaluating `a` share one evaluation of `n`.
  CHECK(isInteger(engine.query(a->getResult(0)), 7));
  CHECK(behaviors.at(n).evaluations == 1);
  // The memo is per query, so hashing `a` again evaluates `n` again.
  CHECK(isInteger(engine.query(a->getResult(0)), 7));
  CHECK(behaviors.at(n).evaluations == 2);
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
  Operation *ignoring = f.op(1, operand->getResult(0));
  behaviors.at(ignoring).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 8)};
  };
  // `ignoring` never queries its operand, so it costs only its own step.
  CHECK(isInteger(oneStep.query(ignoring->getResult(0)), 8));
  CHECK(behaviors.at(operand).evaluations == 0);
  // The step is charged before `evaluate` runs, so `user` runs and its operand
  // exhausts.
  Operation *user = f.op(1, operand->getResult(0));
  CHECK(oneStep.query(user->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(user).evaluations == 1);
  CHECK(behaviors.at(operand).evaluations == 0);
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
    CHECK(!answer.getValue()->isCached());
    CHECK(outcome == CacheOutcome::None);
  }
  CHECK(behaviors.at(small).hashes == 0);
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
  CHECK(behaviors.at(large).hashes == 0);
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
  CHECK(behaviors.at(small).hashes == 0);
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
  semanticKey(a);
  // Eligibility, hashing and evaluating `a` share one evaluation of `n`.
  CHECK(isInteger(engine.query(a->getResult(0)), 7));
  CHECK(behaviors.at(a).hashes == 1);
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
    CHECK(behaviors.at(op).hashes == 0);
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
  CHECK(behaviors.at(op).hashes == 0);
  CHECK(behaviors.at(op).evaluations == 0);
  // `op` must have left the active set on the early return.
  behaviors.at(op).cacheable = Behavior().cacheable;
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
}

void testUncacheableOpsCanBeKeyed() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  Operation *x = f.op(1, f.constant(9));
  Operation *y = f.op(1, f.constant(9));
  Operation *z = f.op(1, f.constant(8));
  for (Operation *op : {x, y, z}) {
    uncacheable(op);
    semanticKey(op);
  }
  // `outer` keys them as an op comparing the ops in its region would.
  Operation *outer = f.op();
  behaviors.at(outer).evaluate = [x, y, z](Operation *, EvalScope &scope) {
    auto key = cast<EvaluableOpInterface>(x);
    auto hash = key.getHash(scope);
    auto otherHash = cast<EvaluableOpInterface>(y).getHash(scope);
    auto same = key.isEquivalentTo(y, scope);
    auto different = key.isEquivalentTo(z, scope);
    bool keyed = hash.completed() && otherHash.completed() &&
                 same.completed() && different.completed() &&
                 hash.getValue() == otherHash.getValue() && same.getValue() &&
                 !different.getValue();
    return SmallVector<Answer>{integer(scope.getAllocator(), keyed)};
  };
  CHECK(isInteger(engine.query(outer->getResult(0)), 1));
  for (Operation *op : {x, y, z}) {
    CHECK(behaviors.at(op).eligibilityChecks == 0);
    CHECK(behaviors.at(op).evaluations == 0);
  }
}

void testIdentityValueKeys() {
  Fixture f;
  Engine engine(f.evalContext, 1000);
  auto key = llvm::hash_code(123);
  Operation *producer = f.hashedOp(key);
  Operation *equivalent = equivalentTo(f, key, producer);
  opaque(producer);
  opaque(equivalent);
  Operation *a = f.op(1, producer->getResult(0));
  Operation *b = f.op(1, equivalent->getResult(0));
  semanticKey(a);
  semanticKey(b);
  // `producer`'s answer is the instance the cache retained, so `a`'s key is
  // the same in every query.
  CacheOutcome outcome;
  CHECK(isInteger(engine.query(a->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(isInteger(engine.query(a->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Hit);
  // `equivalent` hits `producer`'s entry, so `b`'s key holds the same
  // instance as `a`'s and `b` hits too.
  CHECK(isInteger(engine.query(b->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Hit);
  CHECK(behaviors.at(a).evaluations == 1);
  CHECK(behaviors.at(b).evaluations == 0);
  CHECK(behaviors.at(producer).evaluations == 1);
  CHECK(behaviors.at(equivalent).evaluations == 0);

  // Uncached producers answer with a new instance each, so keys holding equal
  // contents still differ.
  Operation *first = f.op();
  Operation *second = f.op();
  for (Operation *op : {first, second}) {
    opaque(op);
    uncacheable(op);
  }
  Operation *c = f.op(1, first->getResult(0));
  Operation *d = f.op(1, second->getResult(0));
  semanticKey(c);
  semanticKey(d);
  CHECK(isInteger(engine.query(c->getResult(0)), 7));
  CHECK(isInteger(engine.query(d->getResult(0), outcome), 7));
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(behaviors.at(d).evaluations == 1);
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
  CHECK(first.completed() && first.getValue()->isCached());
  CHECK(outcome == CacheOutcome::Miss);
  // Another op with the same attribute and type shares the value.
  Answer shared = noSteps.query(sameI32, outcome);
  CHECK(outcome == CacheOutcome::Hit);
  CHECK(shared.completed() && *shared.getValue() == *first.getValue());
  // The same attribute with another type gets its own value.
  Answer other = noSteps.query(f32, outcome);
  CHECK(outcome == CacheOutcome::Miss);
  CHECK(other.completed() && *other.getValue() != *first.getValue());
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
    CHECK(answer.completed() && !answer.getValue()->isCached());
    CHECK(outcome == CacheOutcome::None);
  }
  CHECK(poison.conversions == 6);
}

}

int main() {
  testSemanticReuseAndBudget();
  testZeroHashCollision();
  testPartialResultsAndLifetime();
  testPartialHitFillsWithoutSearching();
  testMissHashesOnce();
  testSignalsAndOptOut();
  testNestedInsertionDuringLookup();
  testNestedInsertionDuringInsert();
  testInsertResumesSearch();
  testEquivalenceConsumesBudget();
  testResultCountFilter();
  testInsertionAfterEvaluation();
  testCycleThroughEquivalence();
  testCycleThroughHash();
  testCycleThroughOperands();
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
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
