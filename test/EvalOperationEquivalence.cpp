#include "interpreter/EvalOperationEquivalence.h"
#include "interpreter/Dialects/Builtin/BuiltinAttrEval.h"
#include "interpreter/Dialects/Builtin/BuiltinEvalValues.h"
#include "interpreter/Engine.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <functional>
#include <map>
#include <optional>

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

Answer unknown() { return {EvalStatus::Completed, std::nullopt}; }

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
  bool cacheable = true;
  /// Forwards operand 0 when unset.
  std::function<SmallVector<Answer>(Operation *, EvalScope &)> evaluate;
};

std::map<Operation *, Behavior> behaviors;

/// Answers with operand 0's answer, or unknown when there are no operands.
SmallVector<Answer> forward(Operation *op, EvalScope &scope) {
  if (op->getNumOperands() == 0)
    return SmallVector<Answer>(op->getNumResults(), unknown());
  Answer answer = scope.query(op->getOperand(0));
  if (!answer.completed())
    return answer.propagate(op);
  return SmallVector<Answer>(op->getNumResults(), answer);
}

/// This class represents a test model that keeps the default `getHash` and
/// `isEquivalentTo`. Its evaluation and eligibility come from the op's
/// `Behavior`.
template <typename OpT>
struct DefaultModel
    : EvaluableOpInterface::ExternalModel<DefaultModel<OpT>, OpT> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    Behavior &behavior = behaviors[op];
    ++behavior.evaluations;
    return behavior.evaluate ? behavior.evaluate(op, scope)
                             : forward(op, scope);
  }

  EvalResult<bool> isCacheable(Operation *op, EvalScope &) const {
    Behavior &behavior = behaviors[op];
    ++behavior.eligibilityChecks;
    return {EvalStatus::Completed, behavior.cacheable};
  }
};

/// Returns whether `answer` holds an odd integer, or nothing when it is
/// unknown.
std::optional<bool> isOdd(const Answer &answer) {
  const auto &value = answer.getValue();
  if (!value)
    return std::nullopt;
  return cast<IntEvalValue>(*value).getValue()[0];
}

/// This class represents a custom key that reuses `EvalOperationEquivalence`
/// with its own callbacks: known integer operands match when their parities
/// do. Every operand hashes the same, so ops that differ only in operands
/// collide and only `isEquivalentTo` tells them apart.
struct ParityModel
    : EvaluableOpInterface::ExternalModel<ParityModel, arith::XOrIOp> {
  EvalResult<llvm::hash_code> getHash(Operation *op, EvalScope &) const {
    return EvalOperationEquivalence::computeHash(op, [](Value) {
      return EvalResult<llvm::hash_code>{EvalStatus::Completed,
                                         llvm::hash_code(0)};
    });
  }

  EvalResult<bool> isEquivalentTo(Operation *op, Operation *other,
                                  EvalScope &scope) const {
    return EvalOperationEquivalence::isEquivalentTo(
        op, other, [&](Value lhs, Value rhs) -> EvalResult<bool> {
          Answer lhsAnswer = scope.query(lhs);
          if (!lhsAnswer.completed())
            return lhsAnswer.status;
          Answer rhsAnswer = scope.query(rhs);
          if (!rhsAnswer.completed())
            return rhsAnswer.status;
          std::optional<bool> lhsOdd = isOdd(lhsAnswer);
          return {EvalStatus::Completed, lhsOdd && lhsOdd == isOdd(rhsAnswer)};
        });
  }

  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    ++behaviors[op].evaluations;
    return forward(op, scope);
  }
};

DialectRegistry registry() {
  DialectRegistry result;
  result.insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                scf::SCFDialect>();
  registerBuiltinAttrEvalExternalModels(result);
  result.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *) {
    arith::AddIOp::attachInterface<DefaultModel<arith::AddIOp>>(*ctx);
    arith::SubIOp::attachInterface<DefaultModel<arith::SubIOp>>(*ctx);
    arith::XOrIOp::attachInterface<ParityModel>(*ctx);
  });
  result.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *) {
    scf::ExecuteRegionOp::attachInterface<DefaultModel<scf::ExecuteRegionOp>>(
        *ctx);
  });
  return result;
}

/// Parses a module whose operations are named by `NameLoc`s. The defaults
/// ignore locations, so names don't affect keys.
struct Fixture {
  MLIRContext context{registry()};
  EvalContext evalContext{&context};
  OwningOpRef<ModuleOp> module;
  /// An uncacheable op that `inQuery` evaluates.
  Operation *probe;

  explicit Fixture(StringRef source) {
    UnrealizedConversionCastOp::attachInterface<
        DefaultModel<UnrealizedConversionCastOp>>(context);
    registerBuiltinEvalValues(evalContext);
    evalContext.registerEvalValues<OpaqueEvalValue>();
    module = parseSourceString<ModuleOp>(source, &context);
    if (!module) {
      llvm::errs() << "failed to parse the test module\n";
      std::exit(1);
    }
    OpBuilder builder = OpBuilder::atBlockEnd(module->getBody());
    probe = UnrealizedConversionCastOp::create(
        builder, builder.getUnknownLoc(), builder.getI32Type(), ValueRange());
    behaviors[probe].cacheable = false;
  }

  ~Fixture() { behaviors.clear(); }

  Operation *op(StringRef name) {
    Operation *result = nullptr;
    module->walk([&](Operation *op) {
      if (auto loc = dyn_cast<NameLoc>(op->getLoc());
          loc && loc.getName() == name)
        result = op;
    });
    if (!result) {
      llvm::errs() << "no operation named " << name << "\n";
      std::exit(1);
    }
    return result;
  }

  Behavior &behavior(StringRef name) { return behaviors[op(name)]; }

  /// Runs `fn` while the probe is evaluated, since keys can only query
  /// operands inside a query.
  void inQuery(function_ref<void(EvalScope &)> fn) {
    behaviors[probe].evaluate = [fn](Operation *, EvalScope &scope) {
      fn(scope);
      return SmallVector<Answer>{unknown()};
    };
    Engine engine(evalContext, 1000);
    engine.query(probe->getResult(0));
  }

  CacheOutcome query(StringRef name) {
    Engine engine(evalContext, 1000);
    CacheOutcome outcome;
    engine.query(op(name)->getResult(0), outcome);
    return outcome;
  }
};

/// Compares two ops with their models' keys in both directions, checking that
/// the answers agree and that equivalent ops hash equally.
bool equivalent(Fixture &f, StringRef lhsName, StringRef rhsName) {
  auto lhs = cast<EvaluableOpInterface>(f.op(lhsName));
  auto rhs = cast<EvaluableOpInterface>(f.op(rhsName));
  bool result = false;
  f.inQuery([&](EvalScope &scope) {
    auto forward = lhs.isEquivalentTo(rhs.getOperation(), scope);
    auto backward = rhs.isEquivalentTo(lhs.getOperation(), scope);
    CHECK(forward.completed() && backward.completed());
    if (!forward.completed() || !backward.completed())
      return;
    CHECK(*forward.getValue() == *backward.getValue());
    result = *forward.getValue();
    if (!result)
      return;
    auto lhsHash = lhs.getHash(scope);
    auto rhsHash = rhs.getHash(scope);
    CHECK(lhsHash.completed() && rhsHash.completed() &&
          *lhsHash.getValue() == *rhsHash.getValue());
  });
  return result;
}

EvalResult<bool> compare(Fixture &f, StringRef lhs, StringRef rhs) {
  std::optional<EvalResult<bool>> result;
  f.inQuery([&](EvalScope &scope) {
    result =
        cast<EvaluableOpInterface>(f.op(lhs)).isEquivalentTo(f.op(rhs), scope);
  });
  return *result;
}

EvalResult<llvm::hash_code> hash(Fixture &f, StringRef name) {
  std::optional<EvalResult<llvm::hash_code>> result;
  f.inQuery([&](EvalScope &scope) {
    result = cast<EvaluableOpInterface>(f.op(name)).getHash(scope);
  });
  return *result;
}

void testStructure() {
  Fixture f(R"mlir(
    func.func @f() {
      %c2 = arith.constant 2 : i32
      %c3 = arith.constant 3 : i32
      %two = arith.constant 2 : i32
      %a = arith.subi %c2, %c3 : i32 loc("a")
      %b = arith.subi %two, %c3 : i32 loc("b")
      %swapped = arith.subi %c3, %c2 : i32 loc("swapped")
      %flags = arith.subi %c2, %c3 overflow<nsw> : i32 loc("flags")
      %attr = arith.subi %c2, %c3 {tag} : i32 loc("attr")
      %add = arith.addi %c2, %c3 : i32 loc("add")
      %i32 = builtin.unrealized_conversion_cast %c2 : i32 to i32 loc("i32")
      %i64 = builtin.unrealized_conversion_cast %two : i32 to i64 loc("i64")
      return
    }
  )mlir");
  CHECK(equivalent(f, "a", "b"));
  CHECK(!equivalent(f, "a", "swapped"));
  CHECK(!equivalent(f, "a", "flags"));
  CHECK(!equivalent(f, "a", "attr"));
  CHECK(!equivalent(f, "a", "add"));
  CHECK(!equivalent(f, "i32", "i64"));

  CHECK(f.query("a") == CacheOutcome::Miss);
  CHECK(f.query("b") == CacheOutcome::Hit);
  CHECK(f.behavior("b").evaluations == 0);
  for (StringRef name : {"swapped", "flags", "attr"})
    CHECK(f.query(name) == CacheOutcome::Miss);
}

void testCommutative() {
  Fixture f(R"mlir(
    func.func @f() {
      %c2 = arith.constant 2 : i32
      %c3 = arith.constant 3 : i32
      %two = arith.constant 2 : i32
      %a = arith.addi %c2, %c3 : i32 loc("a")
      %b = arith.addi %c3, %two : i32 loc("b")
      %twice = arith.addi %c2, %two : i32 loc("twice")
      return
    }
  )mlir");
  CHECK(equivalent(f, "a", "b"));
  // (2, 2) must not match (2, 3) just because each 2 has an equal operand.
  CHECK(!equivalent(f, "twice", "a"));
  CHECK(f.query("a") == CacheOutcome::Miss);
  CHECK(f.query("b") == CacheOutcome::Hit);
}

void testUnknownFallback() {
  Fixture f(R"mlir(
    func.func @f(%arg0: i32, %arg1: i32) {
      %c2 = arith.constant 2 : i32
      %two = arith.constant 2 : i32
      %a = arith.subi %arg0, %c2 : i32 loc("a")
      %b = arith.subi %arg0, %two : i32 loc("b")
      %c = arith.subi %arg1, %c2 : i32 loc("c")
      %known = arith.subi %two, %c2 : i32 loc("known")
      %u = builtin.unrealized_conversion_cast to i32 loc("u")
      %counted = builtin.unrealized_conversion_cast %c2 : i32 to i32 loc("counted")
      %d = arith.subi %u, %c2 : i32 loc("d")
      %e = arith.subi %counted, %c2 : i32 loc("e")
      return
    }
  )mlir");
  // An unknown operand only matches the same SSA value.
  CHECK(equivalent(f, "a", "b"));
  CHECK(!equivalent(f, "a", "c"));
  CHECK(!equivalent(f, "a", "known"));
  CHECK(f.query("a") == CacheOutcome::Miss);
  CHECK(f.query("b") == CacheOutcome::Hit);
  CHECK(f.query("c") == CacheOutcome::Miss);

  // An unknown left operand decides the comparison without the right one.
  f.behavior("counted").cacheable = false;
  auto result = compare(f, "d", "e");
  CHECK(result.completed() && !*result.getValue());
  CHECK(f.behavior("counted").evaluations == 0);
}

void testIdentityValues() {
  Fixture f(R"mlir(
    func.func @f() {
      %c2 = arith.constant 2 : i32
      %o1 = builtin.unrealized_conversion_cast to i32 loc("o1")
      %o2 = builtin.unrealized_conversion_cast to i32 {other} loc("o2")
      %f1 = builtin.unrealized_conversion_cast %o1 : i32 to i32 loc("f1")
      %f2 = builtin.unrealized_conversion_cast %o1 : i32 to i32 loc("f2")
      %a = arith.subi %f1, %c2 : i32 loc("a")
      %b = arith.subi %f2, %c2 : i32 loc("b")
      %c = arith.subi %o2, %c2 : i32 loc("c")
      return
    }
  )mlir");
  for (StringRef name : {"o1", "o2"})
    f.behavior(name).evaluate = [](Operation *, EvalScope &scope) {
      return SmallVector<Answer>{
          {EvalStatus::Completed, OpaqueEvalValue::get(scope.getAllocator())}};
    };
  for (StringRef name : {"f1", "f2"})
    f.behavior(name).cacheable = false;
  // `f1` and `f2` forward the instance the cache retained for `o1`, while `o2`
  // has its own entry and instance.
  CHECK(equivalent(f, "a", "b"));
  CHECK(!equivalent(f, "a", "c"));
  CHECK(f.query("a") == CacheOutcome::Miss);
  CHECK(f.query("b") == CacheOutcome::Hit);
  CHECK(f.query("c") == CacheOutcome::Miss);
  CHECK(f.behavior("b").evaluations == 0);
  CHECK(f.behavior("o1").evaluations == 1);
}

void testRegions() {
  Fixture f(R"mlir(
    func.func @f() {
      %c2 = arith.constant 2 : i32
      %c3 = arith.constant 3 : i32
      %two = arith.constant 2 : i32
      %r1 = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %c2 : i32 to i32 loc("x1")
        %y = arith.addi %x, %c3 : i32
        scf.yield %y : i32
      } loc("r1")
      %r2 = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %two : i32 to i32 loc("x2")
        %y = arith.addi %c3, %x : i32
        scf.yield %y : i32
      } loc("r2")
      %capturedDiffers = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %c3 : i32 to i32
        %y = arith.addi %x, %c3 : i32
        scf.yield %y : i32
      } loc("capturedDiffers")
      %capturedForInternal = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %c2 : i32 to i32
        %y = arith.addi %c2, %c3 : i32
        scf.yield %y : i32
      } loc("capturedForInternal")
      %yieldDiffers = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %c2 : i32 to i32
        %y = arith.addi %x, %c3 : i32
        scf.yield %x : i32
      } loc("yieldDiffers")
      %extraOp = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %c2 : i32 to i32
        %y = arith.addi %x, %c3 : i32
        %z = arith.addi %x, %c3 : i32
        scf.yield %y : i32
      } loc("extraOp")

      %n1 = scf.execute_region -> i32 {
        %i = scf.execute_region -> i32 {
          %z = arith.subi %c2, %c3 : i32
          scf.yield %z : i32
        }
        scf.yield %i : i32
      } loc("n1")
      %n2 = scf.execute_region -> i32 {
        %i = scf.execute_region -> i32 {
          %z = arith.subi %two, %c3 : i32
          scf.yield %z : i32
        }
        scf.yield %i : i32
      } loc("n2")
      %n3 = scf.execute_region -> i32 {
        %i = scf.execute_region -> i32 {
          %z = arith.subi %c3, %c3 : i32
          scf.yield %z : i32
        }
        scf.yield %i : i32
      } loc("n3")

      %b1 = scf.execute_region -> i32 {
        cf.br ^bb1(%c2 : i32)
      ^bb1(%v: i32):
        scf.yield %v : i32
      } loc("b1")
      %b2 = scf.execute_region -> i32 {
        cf.br ^bb1(%two : i32)
      ^bb1(%v: i32):
        scf.yield %v : i32
      } loc("b2")
      %b3 = scf.execute_region -> i32 {
        cf.br ^bb2(%c2 : i32)
      ^bb1(%v: i32):
        scf.yield %v : i32
      ^bb2(%w: i32):
        scf.yield %w : i32
      } loc("b3")
      %b4 = scf.execute_region -> i32 {
        cf.br ^bb1(%c2 : i32)
      ^bb1(%v: i32):
        scf.yield %v : i32
      ^bb2(%w: i32):
        scf.yield %w : i32
      } loc("b4")

      // `^bb1` uses `%w` before `^bb2`, which dominates it, defines it.
      %d1 = scf.execute_region -> i32 {
        cf.br ^bb2
      ^bb1:
        scf.yield %w : i32
      ^bb2:
        %w = arith.subi %c2, %c3 : i32
        cf.br ^bb1
      } loc("d1")
      %d2 = scf.execute_region -> i32 {
        cf.br ^bb2
      ^bb1:
        scf.yield %w : i32
      ^bb2:
        %w = arith.subi %two, %c3 : i32
        cf.br ^bb1
      } loc("d2")
      return
    }
  )mlir");
  // The nested casts opt out of caching, which doesn't exclude them from
  // their enclosing op's key.
  for (StringRef name : {"x1", "x2"})
    f.behavior(name).cacheable = false;

  CHECK(equivalent(f, "r1", "r2"));
  CHECK(!equivalent(f, "r1", "capturedDiffers"));
  CHECK(!equivalent(f, "r1", "capturedForInternal"));
  CHECK(!equivalent(f, "r1", "yieldDiffers"));
  CHECK(!equivalent(f, "r1", "extraOp"));
  CHECK(equivalent(f, "n1", "n2"));
  CHECK(!equivalent(f, "n1", "n3"));
  CHECK(equivalent(f, "b1", "b2"));
  CHECK(!equivalent(f, "b1", "b3"));
  CHECK(!equivalent(f, "b3", "b4"));
  CHECK(equivalent(f, "d1", "d2"));
  CHECK(!equivalent(f, "d1", "n1"));

  for (auto [first, second] : {std::pair("r1", "r2"), std::pair("n1", "n2"),
                               std::pair("b1", "b2"), std::pair("d1", "d2")}) {
    CHECK(f.query(first) == CacheOutcome::Miss);
    CHECK(f.query(second) == CacheOutcome::Hit);
    CHECK(f.behavior(second).evaluations == 0);
  }
  for (StringRef name : {"x1", "x2"}) {
    CHECK(f.behavior(name).eligibilityChecks == 0);
    CHECK(f.behavior(name).evaluations == 0);
  }
}

void testSignals() {
  for (EvalStatus signal :
       {EvalStatus::Exhausted, EvalStatus::NeedsOrder, EvalStatus::Cycle}) {
    Fixture f(R"mlir(
      func.func @f() {
        %c2 = arith.constant 2 : i32
        %c3 = arith.constant 3 : i32
        %sig = builtin.unrealized_conversion_cast to i32 loc("sig")
        %n1 = builtin.unrealized_conversion_cast %c2 : i32 to i32 loc("n1")
        %n2 = builtin.unrealized_conversion_cast %c2 : i32 to i32 loc("n2")
        %n3 = builtin.unrealized_conversion_cast %c3 : i32 to i32 loc("n3")
        %a = arith.subi %sig, %n1 : i32 loc("a")
        %b = arith.subi %n2, %n3 : i32 loc("b")
        %p = arith.addi %c2, %sig : i32 loc("p")
        %q = arith.addi %c3, %c2 : i32 loc("q")
        %r1 = scf.execute_region -> i32 {
          %x = arith.subi %sig, %c2 : i32
          %y = arith.subi %n1, %c2 : i32
          scf.yield %y : i32
        } loc("r1")
        %r2 = scf.execute_region -> i32 {
          %x = arith.subi %c3, %c2 : i32
          %y = arith.subi %n2, %c2 : i32
          scf.yield %y : i32
        } loc("r2")
        return
      }
    )mlir");
    f.behavior("sig").evaluate = [signal](Operation *, EvalScope &) {
      return SmallVector<Answer>{signal};
    };
    for (StringRef name : {"sig", "n1", "n2", "n3"})
      f.behavior(name).cacheable = false;
    auto unevaluated = [&] {
      return f.behavior("n1").evaluations == 0 &&
             f.behavior("n2").evaluations == 0 &&
             f.behavior("n3").evaluations == 0;
    };

    CHECK(hash(f, "a").status == signal);
    CHECK(compare(f, "a", "b").status == signal);
    CHECK(unevaluated());
    // `%c2` pairs with `%c2`, then `%sig` signals against `%c3`.
    CHECK(hash(f, "p").status == signal);
    CHECK(compare(f, "p", "q").status == signal);
    // `%sig` is captured by the first nested op, so the second's operands are
    // never evaluated.
    CHECK(compare(f, "r1", "r2").status == signal);
    CHECK(unevaluated());
  }
}

void testCustomModel() {
  Fixture f(R"mlir(
    func.func @f() {
      %c2 = arith.constant 2 : i32
      %c3 = arith.constant 3 : i32
      %c4 = arith.constant 4 : i32
      %c5 = arith.constant 5 : i32
      %a = arith.xori %c2, %c3 : i32 loc("a")
      %b = arith.xori %c5, %c4 : i32 loc("b")
      %c = arith.xori %c2, %c4 : i32 loc("c")
      return
    }
  )mlir");
  CHECK(equivalent(f, "a", "b"));
  CHECK(!equivalent(f, "a", "c"));
  auto aHash = hash(f, "a");
  auto cHash = hash(f, "c");
  CHECK(aHash.completed() && cHash.completed() &&
        *aHash.getValue() == *cHash.getValue());
  CHECK(f.query("a") == CacheOutcome::Miss);
  CHECK(f.query("b") == CacheOutcome::Hit);
  // `c` collides with `a`'s key but isn't equivalent, so it gets its own entry.
  CHECK(f.query("c") == CacheOutcome::Miss);
  CHECK(f.behavior("b").evaluations == 0);
  CHECK(f.behavior("c").evaluations == 1);
}

} // namespace

int main() {
  testStructure();
  testCommutative();
  testUnknownFallback();
  testIdentityValues();
  testRegions();
  testSignals();
  testCustomModel();
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
