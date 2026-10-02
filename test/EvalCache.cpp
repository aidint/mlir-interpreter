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
  return {AnswerKind::Known, IntEvalValue::get(allocator, APInt(32, value))};
}

bool isInteger(const Answer &answer, int64_t value) {
  return answer.kind == AnswerKind::Known && answer.value &&
         cast<IntEvalValue>(*answer.value).getValue() == value;
}

struct Behavior {
  unsigned hashes = 0;
  unsigned evaluations = 0;
  bool cacheable = true;
  std::function<CacheKeyResult<llvm::hash_code>(Operation *, Engine &)> hash =
      [](Operation *op, Engine &) { return llvm::hash_value(op); };
  std::function<CacheKeyResult<bool>(Operation *, Operation *, Engine &)>
      equal =
          [](Operation *op, Operation *other, Engine &) { return op == other; };
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
  CacheKeyResult<llvm::hash_code> getHash(Operation *op, Engine &engine) const {
    auto &behavior = behaviors.at(op);
    ++behavior.hashes;
    return behavior.hash(op, engine);
  }

  CacheKeyResult<bool> isEqual(Operation *op, Operation *other,
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
    behaviors.at(result).hash = [key](Operation *, Engine &) { return key; };
    return result;
  }
};

void semanticKey(Operation *op) {
  behaviors.at(op).hash =
      [](Operation *op, Engine &engine) -> CacheKeyResult<llvm::hash_code> {
    Answer answer = engine.queryNested(op->getOperand(0));
    if (answer.kind == AnswerKind::Exhausted ||
        answer.kind == AnswerKind::NeedsOrder)
      return answer.kind;
    if (!answer.value)
      return llvm::hash_value(op->getOperand(0).getAsOpaquePointer());
    return cast<EquatableEvalValueInterface>(*answer.value).hash();
  };
  behaviors.at(op).equal = [](Operation *op, Operation *other,
                              Engine &engine) -> CacheKeyResult<bool> {
    if (op->getName() != other->getName() || other->getNumOperands() != 1 ||
        op->getResultTypes() != other->getResultTypes())
      return false;
    Answer lhs = engine.queryNested(op->getOperand(0));
    if (lhs.kind == AnswerKind::Exhausted || lhs.kind == AnswerKind::NeedsOrder)
      return lhs.kind;
    Answer rhs = engine.queryNested(other->getOperand(0));
    if (rhs.kind == AnswerKind::Exhausted || rhs.kind == AnswerKind::NeedsOrder)
      return rhs.kind;
    if (!lhs.value || !rhs.value)
      return !lhs.value && !rhs.value &&
             op->getOperand(0) == other->getOperand(0);
    return mlir::interpreter::isEqual(*lhs.value, *rhs.value);
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
  CHECK(oneStep.query(first->getResult(0)).kind == AnswerKind::Exhausted);
  CHECK(behaviors.at(first).evaluations == 0);
  CHECK(isInteger(oneStep.query(first->getResult(0)), 7));
  CHECK(behaviors.at(first).evaluations == 1);
  CHECK(isInteger(oneStep.query(second->getResult(0)), 7));
  CHECK(behaviors.at(second).evaluations == 0);

  Engine noSteps(f.evalContext, f.cache, 0);
  CHECK(isInteger(noSteps.query(second->getResult(0)), 7));
  CHECK(behaviors.at(first).evaluations == 1);
}

void testPartialResultsAndLifetime() {
  Fixture f;
  Operation *op = f.op(3);
  behaviors.at(op).evaluate = [](Operation *op, EvalScope &scope) {
    if (behaviors.at(op).evaluations == 1)
      return SmallVector<Answer>{integer(scope.getAllocator(), 5),
                                 {AnswerKind::Unknown, std::nullopt},
                                 {AnswerKind::Exhausted, std::nullopt}};
    return SmallVector<Answer>{integer(scope.getAllocator(), 99),
                               integer(scope.getAllocator(), 99),
                               integer(scope.getAllocator(), 13)};
  };
  Answer first;
  {
    Engine engine(f.evalContext, f.cache, 1);
    CHECK(engine.query(op->getResult(2)).kind == AnswerKind::Exhausted);
    auto found = f.cache.lookup(op, engine);
    auto *entry = std::get<CacheEntry *>(found);
    CHECK(entry->results.size() == 3);
    CHECK(entry->results[0] && isInteger(*entry->results[0], 5));
    CHECK(entry->results[1] && entry->results[1]->kind == AnswerKind::Unknown);
    CHECK(!entry->results[2]);
    first = engine.query(op->getResult(0));
    CHECK(isInteger(first, 5));
    CHECK(engine.query(op->getResult(1)).kind == AnswerKind::Unknown);
    CHECK(behaviors.at(op).evaluations == 1);
    CHECK(isInteger(engine.query(op->getResult(2)), 13));
    CHECK(behaviors.at(op).evaluations == 2);
    CHECK(isInteger(*entry->results[0], 5));
    CHECK(entry->results[1]->kind == AnswerKind::Unknown);
    CHECK(isInteger(*entry->results[2], 13));
    CHECK(first.value->isCached());
    f.cache.clear();
    CHECK(std::holds_alternative<llvm::hash_code>(f.cache.lookup(op, engine)));
  }
  CHECK(isInteger(first, 5));
}

void testPreparedHashAndPromotion() {
  Fixture f;
  Engine engine(f.evalContext, f.cache, 0);
  Operation *op = f.op(3);
  auto result = f.cache.lookup(op, engine);
  CHECK(std::holds_alternative<llvm::hash_code>(result));
  CHECK(behaviors.at(op).hashes == 1);
  CacheEntry *entry;
  {
    EvalValueStorageAllocator transient(f.evalContext, false);
    SmallVector<Answer> answers{integer(transient, 42),
                                {AnswerKind::NeedsOrder, std::nullopt},
                                {AnswerKind::Exhausted, std::nullopt}};
    auto inserted =
        f.cache.insert(op, std::get<llvm::hash_code>(result), answers, engine);
    CHECK(behaviors.at(op).hashes == 1);
    entry = std::get<CacheEntry *>(inserted);
    CHECK(entry->results[0]->value->getImpl() != answers[0].value->getImpl());
    CHECK(!entry->results[1] && !entry->results[2]);
    answers[0] = integer(transient, 99);
    answers[1] = {AnswerKind::Unknown, std::nullopt};
    answers[2] = integer(transient, 13);
    CHECK(std::get<CacheEntry *>(f.cache.insert(op, answers, engine)) == entry);
    CHECK(behaviors.at(op).hashes == 2);
  }
  CHECK(isInteger(*entry->results[0], 42));
  CHECK(entry->results[1]->kind == AnswerKind::Unknown);
  CHECK(isInteger(*entry->results[2], 13));
}

void testSignalsAndOptOut() {
  for (AnswerKind signal : {AnswerKind::Exhausted, AnswerKind::NeedsOrder}) {
    Fixture f;
    Engine engine(f.evalContext, f.cache, 100);
    Operation *op = f.op();
    behaviors.at(op).hash =
        [signal](Operation *, Engine &) -> CacheKeyResult<llvm::hash_code> {
      return signal;
    };
    CHECK(std::get<AnswerKind>(f.cache.lookup(op, engine)) == signal);
    SmallVector<Answer> answers{{AnswerKind::Unknown, std::nullopt}};
    CHECK(std::get<AnswerKind>(f.cache.insert(op, answers, engine)) == signal);
    CHECK(engine.query(op->getResult(0)).kind == signal);
    CHECK(behaviors.at(op).evaluations == 0);

    auto key = llvm::hash_code(123);
    Operation *candidate = f.hashedOp(key);
    f.cache.insert(candidate, key, answers, engine);
    behaviors.at(op).hash = [key](Operation *, Engine &) { return key; };
    behaviors.at(op).equal = [signal](Operation *, Operation *,
                                      Engine &) -> CacheKeyResult<bool> {
      return signal;
    };
    CHECK(std::get<AnswerKind>(f.cache.lookup(op, engine)) == signal);
    CHECK(std::get<AnswerKind>(f.cache.insert(op, key, answers, engine)) ==
          signal);
    CHECK(engine.query(op->getResult(0)).kind == signal);
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
  SmallVector<Answer> unknown{{AnswerKind::Unknown, std::nullopt}};
  Operation *candidate = f.hashedOp(key);
  f.cache.insert(candidate, key, unknown, engine);
  Operation *query = f.hashedOp(key);
  Operation *equivalent = f.hashedOp(key);
  behaviors.at(equivalent).equal = [query](Operation *, Operation *other,
                                           Engine &) { return other == query; };
  bool populated = false;
  CacheEntry *nestedEntry = nullptr;
  behaviors.at(query).equal = [&](Operation *, Operation *other,
                                  Engine &engine) -> CacheKeyResult<bool> {
    if (!populated) {
      populated = true;
      for (unsigned i = 0; i != 256; ++i)
        f.cache.insert(f.hashedOp(llvm::hash_code(1000 + i)),
                       llvm::hash_code(1000 + i), unknown, engine);
      for (unsigned i = 0; i != 32; ++i)
        f.cache.insert(f.hashedOp(key), key, unknown, engine);
      nestedEntry = std::get<CacheEntry *>(
          f.cache.insert(equivalent, key, unknown, engine));
    }
    return other == equivalent;
  };
  CacheEntry *found;
  if (inserting)
    found = std::get<CacheEntry *>(f.cache.insert(query, key, unknown, engine));
  else
    found = std::get<CacheEntry *>(f.cache.lookup(query, engine));
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
                                        Engine &) { return other == op; };
  behaviors.at(op).equal = [equivalent](Operation *, Operation *other,
                                        Engine &) {
    return other == equivalent;
  };
  CacheEntry *nestedEntry = nullptr;
  behaviors.at(op).evaluate = [&](Operation *, EvalScope &scope) {
    SmallVector<Answer> answers{integer(scope.getAllocator(), 7)};
    nestedEntry = std::get<CacheEntry *>(
        f.cache.insert(equivalent, key, answers, engine));
    return answers;
  };
  CHECK(isInteger(engine.query(op->getResult(0)), 7));
  CHECK(behaviors.at(op).hashes == 1);
  CHECK(std::get<CacheEntry *>(f.cache.lookup(op, engine)) == nestedEntry);
}

}

int main() {
  testSemanticReuseAndBudget();
  testPartialResultsAndLifetime();
  testPreparedHashAndPromotion();
  testSignalsAndOptOut();
  testNestedInsertion(false);
  testNestedInsertion(true);
  testInsertionAfterEvaluation();
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
