#include "interpreter/Dialects/Builtin/BuiltinAttrEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Engine.h"
#include "interpreter/Interfaces/EquatableEvalValueInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
  return {EvalStatus::Completed,
          IntEvalValue::get(allocator, APInt(32, value))};
}

bool isInteger(const Answer &answer, int64_t value) {
  const auto &result = answer.getValue();
  return answer.status == EvalStatus::Completed && result &&
         cast<IntEvalValue>(*result).getValue() == value;
}

bool isUnknown(const Answer &answer) {
  return answer.status == EvalStatus::Completed && !answer.getValue();
}

struct Behavior {
  unsigned hashes = 0;
  unsigned evaluations = 0;
  bool cacheable = true;
  std::function<EvalResult<llvm::hash_code>(Operation *, EvalScope &)> hash =
      [](Operation *op, EvalScope &) {
        return EvalResult<llvm::hash_code>{EvalStatus::Completed,
                                           llvm::hash_value(op)};
      };
  std::function<EvalResult<bool>(Operation *, Operation *, EvalScope &)> equal =
      [](Operation *op, Operation *other, EvalScope &) {
        return EvalResult<bool>{EvalStatus::Completed, op == other};
      };
  std::function<SmallVector<Answer>(Operation *, EvalScope &)> evaluate =
      [](Operation *op, EvalScope &scope) {
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

  EvalResult<bool> isEqual(Operation *op, Operation *other,
                           EvalScope &scope) const {
    return behaviors.at(op).equal(op, other, scope);
  }

  SmallVector<Answer> evaluate(Operation *op,
                               ArrayRef<std::optional<EvalValue>>,
                               EvalScope &scope) const {
    auto &behavior = behaviors.at(op);
    ++behavior.evaluations;
    return behavior.evaluate(op, scope);
  }

  bool isCacheable(Operation *op) const { return behaviors.at(op).cacheable; }
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
  EvalCache cache{evalContext};
  OwningOpRef<ModuleOp> module{ModuleOp::create(UnknownLoc::get(&context))};
  OpBuilder builder{module->getBody(), module->getBody()->begin()};

  Fixture() {
    context.getOrLoadDialect<arith::ArithDialect>();
    UnrealizedConversionCastOp::attachInterface<TestModel>(context);
    registerBuiltinEvalValues(evalContext);
  }

  ~Fixture() { behaviors.clear(); }

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

  Operation *hashedOp(llvm::hash_code key, unsigned results = 1) {
    Operation *result = op(results);
    behaviors.at(result).hash = [key](Operation *, EvalScope &) {
      return EvalResult<llvm::hash_code>{EvalStatus::Completed, key};
    };
    return result;
  }
};

void semanticKey(Operation *op) {
  behaviors.at(op).hash = [](Operation *op,
                             EvalScope &scope) -> EvalResult<llvm::hash_code> {
    Answer answer = scope.query(op->getOperand(0));
    if (answer.status != EvalStatus::Completed)
      return {answer.status, std::nullopt};
    if (const auto &value = answer.getValue())
      return {EvalStatus::Completed,
              cast<EquatableEvalValueInterface>(*value).hash()};
    return {EvalStatus::Completed,
            llvm::hash_value(op->getOperand(0).getAsOpaquePointer())};
  };
  behaviors.at(op).equal = [](Operation *op, Operation *other,
                              EvalScope &scope) -> EvalResult<bool> {
    if (op->getName() != other->getName() || other->getNumOperands() != 1 ||
        op->getResultTypes() != other->getResultTypes())
      return {EvalStatus::Completed, false};
    Answer lhs = scope.query(op->getOperand(0));
    if (lhs.status != EvalStatus::Completed)
      return {lhs.status, std::nullopt};
    Answer rhs = scope.query(other->getOperand(0));
    if (rhs.status != EvalStatus::Completed)
      return {rhs.status, std::nullopt};
    if (!lhs.getValue() || !rhs.getValue())
      return {EvalStatus::Completed,
              !lhs.getValue() && !rhs.getValue() &&
                  op->getOperand(0) == other->getOperand(0)};
    return {EvalStatus::Completed,
            mlir::interpreter::isEqual(*lhs.getValue(), *rhs.getValue())};
  };
}

void testSemanticReuseAndBudget() {
  Fixture f;
  Value firstConstant = f.constant(9);
  Value secondConstant = f.constant(9);
  Operation *first = f.op(1, firstConstant);
  Operation *second = f.op(1, secondConstant);
  semanticKey(first);
  semanticKey(second);

  Engine twoSteps(f.evalContext, f.cache, 2);
  CHECK(twoSteps.query(first->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(first).evaluations == 0);

  Engine threeSteps(f.evalContext, f.cache, 3);
  CHECK(isInteger(threeSteps.query(first->getResult(0)), 7));
  CHECK(behaviors.at(first).evaluations == 1);
  CHECK(twoSteps.query(second->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(isInteger(threeSteps.query(second->getResult(0)), 7));
  CHECK(behaviors.at(second).evaluations == 0);
  CHECK(behaviors.at(first).evaluations == 1);
}

void testZeroHashCollision() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 1);
  EvalScope scope(engine);
  Operation *first = f.hashedOp(llvm::hash_code(0));
  Operation *second = f.hashedOp(llvm::hash_code(0));
  behaviors.at(second).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 8)};
  };
  auto comparison = cast<EvaluableOpInterface>(second).isEqual(first, scope);
  CHECK(comparison.status == EvalStatus::Completed);
  CHECK(comparison.getValue().has_value());
  CHECK(!*comparison.getValue());
  CHECK(isInteger(engine.query(first->getResult(0)), 7));
  CHECK(isInteger(engine.query(second->getResult(0)), 8));
  CHECK(behaviors.at(first).evaluations == 1);
  CHECK(behaviors.at(second).evaluations == 1);
  Engine noSteps(f.evalContext, f.cache, 0);
  CHECK(isInteger(noSteps.query(first->getResult(0)), 7));
  CHECK(isInteger(noSteps.query(second->getResult(0)), 8));
}

void testPartialResultsAndLifetime() {
  Fixture f;
  Operation *op = f.op(3);
  behaviors.at(op).evaluate = [](Operation *op, EvalScope &scope) {
    if (behaviors.at(op).evaluations == 1)
      return SmallVector<Answer>{integer(scope.getAllocator(), 5),
                                 {EvalStatus::Completed, std::nullopt},
                                 {EvalStatus::Exhausted, std::nullopt}};
    return SmallVector<Answer>{integer(scope.getAllocator(), 99),
                               integer(scope.getAllocator(), 99),
                               integer(scope.getAllocator(), 13)};
  };
  Answer first;
  {
    Engine engine(f.evalContext, f.cache, 1);
    Answer exhausted = engine.query(op->getResult(2));
    CHECK(exhausted.status == EvalStatus::Exhausted);
    CHECK(!exhausted.getValue());
    first = engine.query(op->getResult(0));
    CHECK(isInteger(first, 5));
    CHECK(isUnknown(engine.query(op->getResult(1))));
    CHECK(behaviors.at(op).evaluations == 1);
    CHECK(isInteger(engine.query(op->getResult(2)), 13));
    CHECK(behaviors.at(op).evaluations == 2);
    CHECK(isInteger(engine.query(op->getResult(0)), 5));
    CHECK(isUnknown(engine.query(op->getResult(1))));
    CHECK(first.getValue()->isCached());
    f.cache.clear();
    CHECK(isInteger(engine.query(op->getResult(0)), 99));
    CHECK(behaviors.at(op).evaluations == 3);
  }
  CHECK(isInteger(first, 5));
}

void testPartialHitFillsWithoutSearching() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 1000);
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key, 2);
  behaviors.at(candidate).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 7),
                               {EvalStatus::NeedsOrder, std::nullopt}};
  };
  CHECK(engine.query(candidate->getResult(1)).status == EvalStatus::NeedsOrder);
  Operation *op = f.hashedOp(key, 2);
  unsigned comparisons = 0;
  behaviors.at(op).equal = [&](Operation *, Operation *other, EvalScope &) {
    ++comparisons;
    return EvalResult<bool>{EvalStatus::Completed, other == candidate};
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
  Engine engine(f.evalContext, f.cache, 1000);
  Operation *op = f.op(3);
  Answer answer = engine.query(op->getResult(0));
  CHECK(isInteger(answer, 7));
  CHECK(answer.getValue()->isCached());
  CHECK(behaviors.at(op).hashes == 1);
}

void testSignalsAndOptOut() {
  for (EvalStatus signal : {EvalStatus::Exhausted, EvalStatus::NeedsOrder}) {
    Fixture f;
    Engine engine(f.evalContext, f.cache, 1000);
    Operation *unhashable = f.op();
    behaviors.at(unhashable).hash =
        [signal](Operation *, EvalScope &) -> EvalResult<llvm::hash_code> {
      return {signal, std::nullopt};
    };
    Answer hashFailure = engine.query(unhashable->getResult(0));
    CHECK(hashFailure.status == signal);
    CHECK(!hashFailure.getValue());
    CHECK(behaviors.at(unhashable).evaluations == 0);

    // An equality signal skips the candidate instead of failing the lookup.
    auto key = llvm::hash_code(123);
    Operation *candidate = f.hashedOp(key);
    CHECK(isInteger(engine.query(candidate->getResult(0)), 7));
    Operation *op = f.hashedOp(key);
    behaviors.at(op).equal = [signal](Operation *, Operation *,
                                      EvalScope &) -> EvalResult<bool> {
      return {signal, std::nullopt};
    };
    Answer skipped = engine.query(op->getResult(0));
    CHECK(isInteger(skipped, 7));
    CHECK(skipped.getValue()->isCached());
    CHECK(behaviors.at(op).evaluations == 1);
    CHECK(isInteger(engine.query(op->getResult(0)), 7));
    CHECK(behaviors.at(op).evaluations == 1);

    // Opting out skips hashing, so the signaling hash is never called.
    behaviors.at(unhashable).cacheable = false;
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
                                        EvalScope &) {
    return EvalResult<bool>{EvalStatus::Completed, other == op};
  };
  behaviors.at(equivalent).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 8)};
  };
  return equivalent;
}

void testNestedInsertionDuringLookup() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 1000);
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
    return EvalResult<bool>{EvalStatus::Completed, other == equivalent};
  };
  CHECK(isInteger(engine.query(query->getResult(0)), 8));
  CHECK(populated);
  CHECK(behaviors.at(query).evaluations == 0);
  CHECK(behaviors.at(equivalent).evaluations == 1);
}

void testNestedInsertionDuringInsert() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 1000);
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
    return EvalResult<bool>{EvalStatus::Completed, other == equivalent};
  };
  CHECK(isInteger(engine.query(op->getResult(0)), 8));
  CHECK(populated);
  CHECK(behaviors.at(op).evaluations == 1);
  CHECK(behaviors.at(equivalent).evaluations == 1);
}

void testInsertResumesSearch() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 1000);
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
    return EvalResult<bool>{EvalStatus::Completed, false};
  };
  Answer answer = engine.query(op->getResult(0));
  CHECK(isInteger(answer, 7));
  CHECK(answer.getValue()->isCached());
  CHECK(comparisons[candidate] == 1);
  CHECK(comparisons[x] == 1);
}

void testEqualityConsumesBudget() {
  Fixture f;
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key);
  Engine large(f.evalContext, f.cache, 1000);
  CHECK(isInteger(large.query(candidate->getResult(0)), 7));
  Operation *n1 = f.op();
  Operation *n2 = f.op();
  behaviors.at(n1).cacheable = false;
  behaviors.at(n2).cacheable = false;
  Operation *op = f.hashedOp(key);
  behaviors.at(op).equal = [n1, n2](Operation *, Operation *,
                                    EvalScope &scope) {
    scope.query(n1->getResult(0));
    scope.query(n2->getResult(0));
    return EvalResult<bool>{EvalStatus::Completed, false};
  };
  // Comparing with `candidate` spends both steps, and skipping it refunds
  // nothing.
  Engine twoSteps(f.evalContext, f.cache, 2);
  CHECK(twoSteps.query(op->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(op).evaluations == 0);
  Engine threeSteps(f.evalContext, f.cache, 3);
  CHECK(isInteger(threeSteps.query(op->getResult(0)), 7));
}

void testResultCountFilter() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 1000);
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key, 2);
  CHECK(isInteger(engine.query(candidate->getResult(0)), 7));
  Operation *op = f.hashedOp(key);
  unsigned comparisons = 0;
  // Deliberately wrong: claims equality with every candidate.
  behaviors.at(op).equal = [&](Operation *, Operation *, EvalScope &) {
    ++comparisons;
    return EvalResult<bool>{EvalStatus::Completed, true};
  };
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
  CHECK(comparisons == 0);
  CHECK(behaviors.at(op).evaluations == 1);
}

void testInsertionAfterEvaluation() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 1000);
  auto key = llvm::hash_code(123);
  Operation *op = f.hashedOp(key);
  Operation *equivalent = equivalentTo(f, key, op);
  behaviors.at(op).equal = [equivalent](Operation *, Operation *other,
                                        EvalScope &) {
    return EvalResult<bool>{EvalStatus::Completed, other == equivalent};
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
  testEqualityConsumesBudget();
  testResultCountFilter();
  testInsertionAfterEvaluation();
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
