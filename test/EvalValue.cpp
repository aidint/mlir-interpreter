#include "interpreter/EvalValue.h"
#include "interpreter/Answer.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/EvalContext.h"
#include "interpreter/EvalValueStorageAllocator.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

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

struct NumberStorage : public mlir::interpreter::detail::EvalValueStorage {
  NumberStorage(int64_t value) : value(value) {}
  int64_t value;
};

/// This class represents a test value that defines neither `hash` nor
/// `isEqual`, so only the same instance matches, whatever its contents.
class OpaqueEvalValue : public EvalValueBase<OpaqueEvalValue, NumberStorage> {
public:
  using Base::Base;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpaqueEvalValue)
};

unsigned collidingComparisons = 0;

/// This class represents a test value whose instances all hash alike but are
/// equal only with equal contents.
class CollidingEvalValue
    : public EvalValueBase<CollidingEvalValue, NumberStorage> {
public:
  using Base::Base;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollidingEvalValue)

  bool isEqual(CollidingEvalValue other) const {
    ++collidingComparisons;
    return getImpl()->value == other.getImpl()->value;
  }

  llvm::hash_code hash() const { return llvm::hash_code(0); }
};

struct Fixture {
  MLIRContext context;
  EvalContext evalContext{&context};
  EvalValueStorageAllocator allocator{evalContext, false};
  EvalValueStorageAllocator cache{evalContext, true};

  Fixture() {
    registerBuiltinEvalValues(evalContext);
    evalContext.registerEvalValues<OpaqueEvalValue, CollidingEvalValue>();
    collidingComparisons = 0;
  }

  EvalValue integer(unsigned width, int64_t value) {
    return IntEvalValue::get(allocator, APInt(width, value));
  }
};

void testIdentityDefaults() {
  Fixture f;
  EvalValue value = OpaqueEvalValue::get(f.allocator, 5);
  EvalValue copy = value;
  EvalValue twin = OpaqueEvalValue::get(f.allocator, 5);
  CHECK(value.isEqual(value));
  CHECK(copy.isEqual(value) && value.isEqual(copy));
  CHECK(copy.hash() == value.hash());
  // Equal contents don't make distinct instances equal.
  CHECK(!value.isEqual(twin) && !twin.isEqual(value));
  // Retaining clones the storage, so the copy is a distinct instance.
  EvalValue retained = f.cache.retain(value);
  CHECK(retained.isCached());
  CHECK(!retained.isEqual(value) && !value.isEqual(retained));
  CHECK(retained.isEqual(f.cache.retain(value)));
}

void testContentEquality() {
  Fixture f;
  EvalValue five = f.integer(32, 5);
  EvalValue other = f.integer(32, 5);
  CHECK(five.isEqual(other) && other.isEqual(five));
  CHECK(five.hash() == other.hash());
  CHECK(!five.isEqual(f.integer(32, 6)));
  CHECK(!five.isEqual(f.integer(64, 5)));
  EvalValue retained = f.cache.retain(five);
  CHECK(retained.isEqual(five) && retained.hash() == five.hash());
  // Handles still compare, hash and key maps by identity.
  CHECK(five != other);
  CHECK(llvm::DenseSet<EvalValue>({five, other}).size() == 2);
}

void testDifferentClasses() {
  Fixture f;
  EvalValue integer = f.integer(32, 5);
  EvalValue opaque = OpaqueEvalValue::get(f.allocator, 5);
  EvalValue colliding = CollidingEvalValue::get(f.allocator, 5);
  CHECK(!integer.isEqual(opaque) && !opaque.isEqual(integer));
  CHECK(!integer.isEqual(colliding) && !colliding.isEqual(integer));
  CHECK(!opaque.isEqual(colliding) && !colliding.isEqual(opaque));
  // Classes differ, so neither class's `isEqual` runs.
  CHECK(collidingComparisons == 0);
}

void testHashCollisions() {
  Fixture f;
  EvalValue one = CollidingEvalValue::get(f.allocator, 1);
  EvalValue two = CollidingEvalValue::get(f.allocator, 2);
  EvalValue alsoOne = CollidingEvalValue::get(f.allocator, 1);
  CHECK(one.hash() == two.hash());
  CHECK(!one.isEqual(two));
  CHECK(one.isEqual(alsoOne));
  CHECK(collidingComparisons == 2);
  // An instance equals itself without calling the class.
  CHECK(one.isEqual(one));
  CHECK(collidingComparisons == 2);
}

void testAnswerStates() {
  Fixture f;
  CHECK(Answer::unknown().isUnknown());
  // Unknown is a completed answer holding null, so a null value converts to it.
  CHECK(Answer(EvalValue()).isUnknown());
  CHECK(!Answer(f.integer(32, 5)).isUnknown());
  CHECK(!Answer(EvalStatus::Exhausted).isUnknown());
}

// A status and a handle, with no optional's flag: 16 bytes on 64-bit hosts.
static_assert(sizeof(Answer) == 2 * sizeof(void *));

} // namespace

int main() {
  testIdentityDefaults();
  testContentEquality();
  testDifferentClasses();
  testHashCollisions();
  testAnswerStates();
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
