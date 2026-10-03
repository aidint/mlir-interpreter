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
  std::function<EvalResult<llvm::hash_code>(Operation *, Engine &)> hash =
      [](Operation *op, Engine &) {
        return EvalResult<llvm::hash_code>{EvalStatus::Completed,
                                           llvm::hash_value(op)};
      };
  std::function<EvalResult<bool>(Operation *, Operation *, Engine &)> equal =
      [](Operation *op, Operation *other, Engine &) {
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
  EvalResult<llvm::hash_code> getHash(Operation *op, Engine &engine) const {
    auto &behavior = behaviors.at(op);
    ++behavior.hashes;
    return behavior.hash(op, engine);
  }

  EvalResult<bool> isEqual(Operation *op, Operation *other,
                           Engine &engine) const {
    return behaviors.at(op).equal(op, other, engine);
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

  Operation *hashedOp(llvm::hash_code key) {
    Operation *result = op();
    behaviors.at(result).hash = [key](Operation *, Engine &) {
      return EvalResult<llvm::hash_code>{EvalStatus::Completed, key};
    };
    return result;
  }
};

void semanticKey(Operation *op) {
  behaviors.at(op).hash = [](Operation *op,
                             Engine &engine) -> EvalResult<llvm::hash_code> {
    Answer answer = engine.queryNested(op->getOperand(0));
    if (answer.status != EvalStatus::Completed)
      return {answer.status, std::nullopt};
    if (const auto &value = answer.getValue())
      return {EvalStatus::Completed,
              cast<EquatableEvalValueInterface>(*value).hash()};
    return {EvalStatus::Completed,
            llvm::hash_value(op->getOperand(0).getAsOpaquePointer())};
  };
  behaviors.at(op).equal = [](Operation *op, Operation *other,
                              Engine &engine) -> EvalResult<bool> {
    if (op->getName() != other->getName() || other->getNumOperands() != 1 ||
        op->getResultTypes() != other->getResultTypes())
      return {EvalStatus::Completed, false};
    Answer lhs = engine.queryNested(op->getOperand(0));
    if (lhs.status != EvalStatus::Completed)
      return {lhs.status, std::nullopt};
    Answer rhs = engine.queryNested(other->getOperand(0));
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

  Engine oneStep(f.evalContext, f.cache, 1);
  CHECK(oneStep.query(first->getResult(0)).status == EvalStatus::Exhausted);
  CHECK(behaviors.at(first).evaluations == 0);
  CHECK(isInteger(oneStep.query(first->getResult(0)), 7));
  CHECK(behaviors.at(first).evaluations == 1);
  CHECK(isInteger(oneStep.query(second->getResult(0)), 7));
  CHECK(behaviors.at(second).evaluations == 0);

  Engine noSteps(f.evalContext, f.cache, 0);
  CHECK(isInteger(noSteps.query(second->getResult(0)), 7));
  CHECK(behaviors.at(first).evaluations == 1);
}

void testZeroHashCollision() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 1);
  Operation *first = f.hashedOp(llvm::hash_code(0));
  Operation *second = f.hashedOp(llvm::hash_code(0));
  behaviors.at(second).evaluate = [](Operation *, EvalScope &scope) {
    return SmallVector<Answer>{integer(scope.getAllocator(), 8)};
  };
  auto comparison =
      cast<EvaluableOpInterface>(second).isEqual(first, engine);
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
    auto *entry = f.cache.lookup(op, engine).getValue()->entry;
    CHECK(entry->results.size() == 3);
    CHECK(entry->results[0] && isInteger(*entry->results[0], 5));
    CHECK(entry->results[1] && isUnknown(*entry->results[1]));
    CHECK(!entry->results[2]);
    first = engine.query(op->getResult(0));
    CHECK(isInteger(first, 5));
    CHECK(isUnknown(engine.query(op->getResult(1))));
    CHECK(behaviors.at(op).evaluations == 1);
    CHECK(isInteger(engine.query(op->getResult(2)), 13));
    CHECK(behaviors.at(op).evaluations == 2);
    CHECK(isInteger(*entry->results[0], 5));
    CHECK(isUnknown(*entry->results[1]));
    CHECK(isInteger(*entry->results[2], 13));
    CHECK(first.getValue()->isCached());
    f.cache.clear();
    CHECK(!f.cache.lookup(op, engine).getValue()->entry);
  }
  CHECK(isInteger(first, 5));
}

void testPartialHitFillsWithoutSearching() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 1);
  auto key = llvm::hash_code(123);
  Operation *candidate = f.hashedOp(key);
  SmallVector<Answer> pending{{EvalStatus::NeedsOrder, std::nullopt}};
  CacheEntry *entry =
      f.cache.insert(candidate, key, pending, engine).getValue().value();
  Operation *op = f.hashedOp(key);
  unsigned comparisons = 0;
  behaviors.at(op).equal = [&](Operation *, Operation *other, Engine &) {
    ++comparisons;
    return EvalResult<bool>{EvalStatus::Completed, other == candidate};
  };
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
  CHECK(behaviors.at(op).hashes == 1);
  CHECK(comparisons == 1);
  CHECK(isInteger(*entry->results[0], 7));
}

void testPreparedHashAndPromotion() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 0);
  Operation *op = f.op(3);
  auto result = f.cache.lookup(op, engine);
  CHECK(!result.getValue()->entry);
  CHECK(behaviors.at(op).hashes == 1);
  CacheEntry *entry;
  {
    EvalValueStorageAllocator transient(f.evalContext, false);
    SmallVector<Answer> answers{integer(transient, 42),
                                {EvalStatus::NeedsOrder, std::nullopt},
                                {EvalStatus::Exhausted, std::nullopt}};
    auto inserted = f.cache.insert(op, result.getValue()->key, answers, engine);
    CHECK(behaviors.at(op).hashes == 1);
    entry = inserted.getValue().value();
    CHECK(entry->results[0]->getValue()->getImpl() !=
          answers[0].getValue()->getImpl());
    CHECK(!entry->results[1] && !entry->results[2]);
    answers[0] = integer(transient, 99);
    answers[1] = {EvalStatus::Completed, std::nullopt};
    answers[2] = integer(transient, 13);
    CHECK(f.cache.insert(op, answers, engine).getValue().value() == entry);
    CHECK(behaviors.at(op).hashes == 2);
  }
  CHECK(isInteger(*entry->results[0], 42));
  CHECK(isUnknown(*entry->results[1]));
  CHECK(isInteger(*entry->results[2], 13));
}

void testSignalsAndOptOut() {
  for (EvalStatus signal : {EvalStatus::Exhausted, EvalStatus::NeedsOrder}) {
    Fixture f;
    Engine engine(f.evalContext, f.cache, 100);
    Operation *op = f.op();
    behaviors.at(op).hash = [signal](Operation *,
                                     Engine &) -> EvalResult<llvm::hash_code> {
      return {signal, std::nullopt};
    };
    CHECK(f.cache.lookup(op, engine).status == signal);
    SmallVector<Answer> answers{{EvalStatus::Completed, std::nullopt}};
    CHECK(f.cache.insert(op, answers, engine).status == signal);
    Answer hashFailure = engine.query(op->getResult(0));
    CHECK(hashFailure.status == signal);
    CHECK(!hashFailure.getValue());
    CHECK(behaviors.at(op).evaluations == 0);

    auto key = llvm::hash_code(123);
    Operation *candidate = f.hashedOp(key);
    f.cache.insert(candidate, key, answers, engine);
    behaviors.at(op).hash = [key](Operation *, Engine &) {
      return EvalResult<llvm::hash_code>{EvalStatus::Completed, key};
    };
    behaviors.at(op).equal = [signal](Operation *, Operation *,
                                      Engine &) -> EvalResult<bool> {
      return {signal, std::nullopt};
    };
    CHECK(f.cache.lookup(op, engine).status == signal);
    CHECK(f.cache.insert(op, key, answers, engine).status == signal);
    Answer equalityFailure = engine.query(op->getResult(0));
    CHECK(equalityFailure.status == signal);
    CHECK(!equalityFailure.getValue());
    CHECK(behaviors.at(op).evaluations == 0);

    behaviors.at(op).cacheable = false;
    CHECK(isInteger(engine.query(op->getResult(0)), 7));
    CHECK(isInteger(engine.query(op->getResult(0)), 7));
    CHECK(behaviors.at(op).evaluations == 2);
  }
}

void testNestedInsertion(bool inserting) {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 10);
  auto key = llvm::hash_code(123);
  SmallVector<Answer> unknown{{EvalStatus::Completed, std::nullopt}};
  Operation *candidate = f.hashedOp(key);
  f.cache.insert(candidate, key, unknown, engine);
  Operation *query = f.hashedOp(key);
  Operation *equivalent = f.hashedOp(key);
  behaviors.at(equivalent).equal = [query](Operation *, Operation *other,
                                           Engine &) {
    return EvalResult<bool>{EvalStatus::Completed, other == query};
  };
  bool populated = false;
  CacheEntry *nestedEntry = nullptr;
  behaviors.at(query).equal = [&](Operation *, Operation *other,
                                  Engine &engine) -> EvalResult<bool> {
    if (!populated) {
      populated = true;
      for (unsigned i = 0; i != 256; ++i)
        f.cache.insert(f.hashedOp(llvm::hash_code(1000 + i)),
                       llvm::hash_code(1000 + i), unknown, engine);
      for (unsigned i = 0; i != 32; ++i)
        f.cache.insert(f.hashedOp(key), key, unknown, engine);
      nestedEntry =
          f.cache.insert(equivalent, key, unknown, engine).getValue().value();
    }
    return EvalResult<bool>{EvalStatus::Completed, other == equivalent};
  };
  CacheEntry *found;
  if (inserting)
    found = f.cache.insert(query, key, unknown, engine).getValue().value();
  else
    found = f.cache.lookup(query, engine).getValue()->entry;
  CHECK(populated);
  CHECK(found == nestedEntry);
  CHECK(found->op == equivalent);
}

void testInsertionAfterEvaluation() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 10);
  auto key = llvm::hash_code(123);
  Operation *op = f.hashedOp(key);
  Operation *equivalent = f.hashedOp(key);
  behaviors.at(equivalent).equal = [op](Operation *, Operation *other,
                                        Engine &) {
    return EvalResult<bool>{EvalStatus::Completed, other == op};
  };
  behaviors.at(op).equal = [equivalent](Operation *, Operation *other,
                                        Engine &) {
    return EvalResult<bool>{EvalStatus::Completed, other == equivalent};
  };
  CacheEntry *nestedEntry = nullptr;
  behaviors.at(op).evaluate = [&](Operation *, EvalScope &scope) {
    SmallVector<Answer> answers{integer(scope.getAllocator(), 7)};
    nestedEntry =
        f.cache.insert(equivalent, key, answers, engine).getValue().value();
    return answers;
  };
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
  CHECK(behaviors.at(op).hashes == 1);
  CHECK(f.cache.lookup(op, engine).getValue()->entry == nestedEntry);
}

}

int main() {
  testSemanticReuseAndBudget();
  testZeroHashCollision();
  testPartialResultsAndLifetime();
  testPartialHitFillsWithoutSearching();
  testPreparedHashAndPromotion();
  testSignalsAndOptOut();
  testNestedInsertion(false);
  testNestedInsertion(true);
  testInsertionAfterEvaluation();
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
