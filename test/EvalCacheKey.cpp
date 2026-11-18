#include "interpreter/EvalCacheKey.h"
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <map>

using namespace mlir;
using namespace mlir::interpreter;
using namespace mlir::interpreter::detail;

namespace {

unsigned failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      llvm::errs() << "FAIL " << __LINE__ << ": " << #condition << "\n";       \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

/// This class represents a snapshot and the value of each of its input slots.
struct Snapshot {
  const EvalCacheOp *op;
  SmallVector<Value> outerValues;
};

struct Behavior {
  unsigned evaluations = 0;
  /// Forwards operand 0 when unset.
  std::function<SmallVector<Answer>(Operation *, EvalScope &)> evaluate;
};

std::map<Operation *, Behavior> behaviors;

/// Answers with operand 0's answer, or unknown when there are no operands.
SmallVector<Answer> forward(Operation *op, EvalScope &scope) {
  if (op->getNumOperands() == 0)
    return SmallVector<Answer>(op->getNumResults(), Answer::unknown());
  Answer answer = scope.query(op->getOperand(0));
  if (!answer.completed())
    return answer.propagate(op);
  return SmallVector<Answer>(op->getNumResults(), answer);
}

/// This class represents an uncacheable test model whose evaluation comes from
/// the op's `Behavior`.
struct CastModel
    : EvaluableOpInterface::ExternalModel<CastModel,
                                          UnrealizedConversionCastOp> {
  SmallVector<Answer> evaluate(Operation *op, EvalScope &scope) const {
    Behavior &behavior = behaviors[op];
    ++behavior.evaluations;
    return behavior.evaluate ? behavior.evaluate(op, scope)
                             : forward(op, scope);
  }

  EvalResult<bool> isCacheable(Operation *, EvalScope &) const { return false; }
};

DialectRegistry registry() {
  DialectRegistry result;
  result.insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                scf::SCFDialect>();
  registerBuiltinAttrEvalExternalModels(result);
  return result;
}

using KeyInfo = llvm::DenseMapInfo<CacheKey>;

/// Returns whether two keys are equal, checking that equality is symmetric and
/// that equal keys hash equally.
bool keysEqual(const CacheKey &lhs, const CacheKey &rhs) {
  bool result = KeyInfo::isEqual(lhs, rhs);
  CHECK(result == KeyInfo::isEqual(rhs, lhs));
  if (result)
    CHECK(KeyInfo::getHashValue(lhs) == KeyInfo::getHashValue(rhs));
  return result;
}

/// This class parses a module whose operations are named by `NameLoc`s, and
/// snapshots them or builds their keys. Locations aren't part of a snapshot,
/// so names don't change it.
struct Fixture {
  MLIRContext context{registry()};
  EvalContext evalContext{&context};
  OwningOpRef<ModuleOp> module;
  llvm::BumpPtrAllocator allocator;
  /// An uncacheable op that `inQuery` evaluates.
  Operation *probe;

  /// Graph regions of unregistered ops are only accepted without verification.
  explicit Fixture(StringRef source, bool verify = true) {
    context.allowUnregisteredDialects();
    UnrealizedConversionCastOp::attachInterface<CastModel>(context);
    registerBuiltinEvalValues(evalContext);
    module =
        parseSourceString<ModuleOp>(source, ParserConfig(&context, verify));
    if (!module) {
      llvm::errs() << "failed to parse the test module\n";
      std::exit(1);
    }
    OpBuilder builder = OpBuilder::atBlockEnd(module->getBody());
    probe = UnrealizedConversionCastOp::create(
        builder, builder.getUnknownLoc(), builder.getI32Type(), ValueRange());
  }

  ~Fixture() { behaviors.clear(); }

  Behavior &behavior(StringRef name) { return behaviors[op(name)]; }

  /// Runs `fn` while the probe is evaluated, since keys can only query their
  /// inputs inside a query.
  void inQuery(function_ref<void(EvalScope &)> fn) {
    behaviors[probe].evaluate = [fn](Operation *, EvalScope &scope) {
      fn(scope);
      return SmallVector<Answer>{Answer::unknown()};
    };
    Engine engine(evalContext, 1000);
    engine.query(probe->getResult(0));
  }

  /// Builds the key of `name` in `scope`, or exits on a signal.
  CacheKey key(StringRef name, EvalScope &scope) {
    EvalResult<CacheKey> result = buildCacheKey(op(name), scope, allocator);
    if (!result.completed()) {
      llvm::errs() << "building the key of " << name << " signalled\n";
      std::exit(1);
    }
    return result.getValue();
  }

  /// Builds the keys of two ops in one query, and returns whether they are
  /// equal.
  bool sameKey(StringRef lhs, StringRef rhs) {
    bool result = false;
    inQuery([&](EvalScope &scope) {
      result = keysEqual(key(lhs, scope), key(rhs, scope));
    });
    return result;
  }

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

  /// Returns the first result of each op in `names`.
  SmallVector<Value> values(std::initializer_list<StringRef> names) {
    return llvm::map_to_vector(
        names, [&](StringRef name) -> Value { return op(name)->getResult(0); });
  }

  Snapshot snapshot(StringRef name) {
    CacheKeyBuilder builder(allocator);
    const EvalCacheOp *snapshot = builder.build(op(name));
    return {snapshot, llvm::to_vector(builder.getOuterValues())};
  }

  /// Returns whether two ops have equal snapshots, checking that equality is
  /// symmetric and that equal snapshots hash equally.
  bool same(StringRef lhsName, StringRef rhsName) {
    const EvalCacheOp &lhs = *snapshot(lhsName).op;
    const EvalCacheOp &rhs = *snapshot(rhsName).op;
    bool result = lhs == rhs;
    CHECK(result == (rhs == lhs));
    if (result)
      CHECK(hash_value(lhs) == hash_value(rhs));
    return result;
  }
};

OperandRef inner(unsigned index) { return OperandRef::getInner(index); }
OperandRef input(unsigned index) { return OperandRef::getInput(index); }

bool refsAre(ArrayRef<OperandRef> refs,
             std::initializer_list<OperandRef> expected) {
  return refs == ArrayRef<OperandRef>(expected);
}

void testStructure() {
  Fixture f(R"mlir(
    func.func @f() {
      %c2 = arith.constant 2 : i32 loc("c2")
      %c3 = arith.constant 3 : i32 loc("c3")
      %two = arith.constant 2 : i32 loc("two")
      %a = arith.subi %c2, %c3 : i32 loc("a")
      %b = arith.subi %two, %c3 : i32 loc("b")
      %swapped = arith.subi %c3, %c2 : i32 loc("swapped")
      %flags = arith.subi %c2, %c3 overflow<nsw> : i32 loc("flags")
      %attr = arith.subi %c2, %c3 {tag} : i32 loc("attr")
      %add = arith.addi %c2, %c3 : i32 loc("add")
      %i32 = builtin.unrealized_conversion_cast %c2 : i32 to i32 loc("i32")
      %i64 = builtin.unrealized_conversion_cast %c2 : i32 to i64 loc("i64")
      return
    }
  )mlir");
  CHECK(f.same("a", "b"));
  CHECK(refsAre(f.snapshot("a").op->operands, {input(0), input(1)}));
  // Swapping outer operands only swaps the values of the input slots.
  CHECK(f.same("a", "swapped"));
  CHECK(f.snapshot("a").outerValues == f.values({"c2", "c3"}));
  CHECK(f.snapshot("swapped").outerValues == f.values({"c3", "c2"}));
  CHECK(!f.same("a", "flags"));
  CHECK(!f.same("a", "attr"));
  CHECK(!f.same("a", "add"));
  CHECK(!f.same("i32", "i64"));
}

void testRegions() {
  Fixture f(R"mlir(
    func.func @f() {
      %c2 = arith.constant 2 : i32 loc("c2")
      %c3 = arith.constant 3 : i32 loc("c3")
      %two = arith.constant 2 : i32 loc("two")
      %r1 = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %c2 : i32 to i32
        %y = arith.addi %x, %c3 : i32
        scf.yield %y : i32
      } loc("r1")
      %r2 = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %two : i32 to i32
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
      return
    }
  )mlir");
  CHECK(f.same("r1", "capturedDiffers"));
  CHECK(f.snapshot("r1").outerValues == f.values({"c2", "c3"}));
  CHECK(f.snapshot("capturedDiffers").outerValues == f.values({"c3", "c3"}));
  CHECK(!f.same("r1", "capturedForInternal"));
  CHECK(!f.same("r1", "yieldDiffers"));
  CHECK(!f.same("r1", "extraOp"));
  // Operands are compared in order, also those of a commutative op.
  CHECK(!f.same("r1", "r2"));

  CHECK(f.same("n1", "n2"));
  CHECK(f.same("n1", "n3"));
  CHECK(f.snapshot("n3").outerValues == f.values({"c3", "c3"}));
  CHECK(!f.same("n1", "r1"));
}

void testBlocks() {
  Fixture f(R"mlir(
    func.func @f() {
      %c2 = arith.constant 2 : i32 loc("c2")
      %c3 = arith.constant 3 : i32 loc("c3")
      %two = arith.constant 2 : i32 loc("two")
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
      %d3 = scf.execute_region -> i32 {
        cf.br ^bb2
      ^bb1:
        scf.yield %c2 : i32
      ^bb2:
        %w = arith.subi %two, %c3 : i32
        cf.br ^bb1
      } loc("d3")
      return
    }
  )mlir");
  CHECK(f.same("b1", "b2"));
  const EvalCacheRegion &b1 = f.snapshot("b1").op->regions[0];
  CHECK(b1.blocks.size() == 2);
  CHECK(b1.blocks[0].ops[0].successors == ArrayRef<unsigned>{1});
  CHECK(b1.blocks[1].argTypes.size() == 1);
  // `%v` is numbered after `%b1`.
  CHECK(refsAre(b1.blocks[1].ops[0].operands, {inner(1)}));
  CHECK(!f.same("b1", "b3"));
  CHECK(!f.same("b3", "b4"));

  CHECK(f.same("d1", "d2"));
  // The yield's `%w` was parked, then found to be defined inside.
  Snapshot d1 = f.snapshot("d1");
  CHECK(refsAre(d1.op->regions[0].blocks[1].ops[0].operands, {inner(1)}));
  CHECK(d1.outerValues == f.values({"c2", "c3"}));
  CHECK(!f.same("d1", "d3"));
}

void testLoop() {
  Fixture f(R"mlir(
    func.func @f(%x: index) {
      %c0 = arith.constant 0 : index loc("c0")
      %c1 = arith.constant 1 : index loc("c1")
      %c4 = arith.constant 4 : index loc("c4")
      %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %c0) -> (index) {
        %n = arith.addi %a, %x : index
        scf.yield %n : index
      } loc("r")
      return
    }
  )mlir");
  Snapshot r = f.snapshot("r");
  CHECK(refsAre(r.op->operands, {input(0), input(1), input(2), input(3)}));
  const EvalCacheBlock &body = r.op->regions[0].blocks[0];
  CHECK(body.argTypes.size() == 2);
  CHECK(refsAre(body.ops[0].operands, {inner(2), input(4)}));
  CHECK(refsAre(body.ops[1].operands, {inner(3)}));
  // One slot per use, so `%c0` takes two.
  Value x = f.op("r")->getParentOfType<func::FuncOp>().getArgument(0);
  SmallVector<Value> expected = f.values({"c0", "c4", "c1", "c0"});
  expected.push_back(x);
  CHECK(r.outerValues == expected);
}

void testGraphRegions() {
  Fixture f(R"mlir(
    %x = "test.x"() : () -> i32 loc("x")
    %y = "test.y"() : () -> i32 loc("y")
    %self = "test.op"(%self) : (i32) -> i32 loc("self")
    %r = "test.root"(%x) ({
      %a = "test.op"(%b) : (i32) -> i32
      %b = "test.op"(%y) : (i32) -> i32
    }) : (i32) -> i32 loc("root")
  )mlir",
            /*verify=*/false);
  Snapshot self = f.snapshot("self");
  CHECK(refsAre(self.op->operands, {inner(0)}));
  CHECK(self.outerValues.empty());

  Snapshot root = f.snapshot("root");
  CHECK(refsAre(root.op->operands, {input(0)}));
  ArrayRef<EvalCacheOp> ops = root.op->regions[0].blocks[0].ops;
  CHECK(refsAre(ops[0].operands, {inner(2)}));
  CHECK(refsAre(ops[1].operands, {input(1)}));
  CHECK(root.outerValues == f.values({"x", "y"}));
}

void testInputs() {
  Fixture f(R"mlir(
    func.func @f(%arg0: i32, %arg1: i32) {
      %c2 = arith.constant 2 : i32
      %c3 = arith.constant 3 : i32
      %two = arith.constant 2 : i32
      %a = arith.subi %c2, %c3 : i32 loc("a")
      %b = arith.subi %two, %c3 : i32 loc("b")
      %swapped = arith.subi %c3, %c2 : i32 loc("swapped")
      %u0 = arith.subi %arg0, %c2 : i32 loc("u0")
      %u1 = arith.subi %arg1, %two : i32 loc("u1")
      %r1 = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %c2 : i32 to i32
        %y = arith.addi %x, %c3 : i32
        scf.yield %y : i32
      } loc("r1")
      %r2 = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %two : i32 to i32
        %y = arith.addi %x, %c3 : i32
        scf.yield %y : i32
      } loc("r2")
      %capturedDiffers = scf.execute_region -> i32 {
        %x = builtin.unrealized_conversion_cast %c3 : i32 to i32
        %y = arith.addi %x, %c3 : i32
        scf.yield %y : i32
      } loc("capturedDiffers")

      %i64 = arith.constant 2 : i64
      %idx = arith.constant 2 : index
      %fromI64 = builtin.unrealized_conversion_cast %i64 : i64 to i32 loc("fromI64")
      %fromIndex = builtin.unrealized_conversion_cast %idx : index to i32 loc("fromIndex")
      return
    }
  )mlir");
  CHECK(f.sameKey("a", "b"));
  CHECK(!f.sameKey("a", "swapped"));
  CHECK(f.sameKey("r1", "r2"));
  CHECK(!f.sameKey("r1", "capturedDiffers"));

  // Unknown matches only unknown, whichever value it came from.
  CHECK(f.sameKey("u0", "u1"));
  CHECK(!f.sameKey("u0", "swapped"));
  f.inQuery([&](EvalScope &scope) {
    CacheKey u0 = f.key("u0", scope);
    CHECK(!u0.inputs[0] && u0.inputs[1]);
  });

  // Both inputs are 64-bit 2s, so only their types tell them apart.
  CHECK(!f.sameKey("fromI64", "fromIndex"));
}

void testSignals() {
  for (EvalStatus signal :
       {EvalStatus::Exhausted, EvalStatus::NeedsOrder, EvalStatus::Cycle}) {
    Fixture f(R"mlir(
      func.func @f() {
        %c2 = arith.constant 2 : i32
        %sig = builtin.unrealized_conversion_cast to i32 loc("sig")
        %n = builtin.unrealized_conversion_cast %c2 : i32 to i32 loc("n")
        %a = arith.subi %sig, %n : i32 loc("a")
        %b = arith.subi %n, %sig : i32 loc("b")
        %r = scf.execute_region -> i32 {
          %x = arith.subi %sig, %c2 : i32
          %y = arith.subi %n, %c2 : i32
          scf.yield %y : i32
        } loc("r")
        return
      }
    )mlir");
    f.behavior("sig").evaluate = [signal](Operation *, EvalScope &) {
      return SmallVector<Answer>{signal};
    };
    auto status = [&](StringRef name) {
      EvalStatus result = EvalStatus::Completed;
      f.inQuery([&](EvalScope &scope) {
        result = buildCacheKey(f.op(name), scope, f.allocator).status;
      });
      return result;
    };

    CHECK(status("a") == signal);
    CHECK(status("r") == signal);
    CHECK(f.behavior("n").evaluations == 0);
    // Inputs before the signal are evaluated.
    CHECK(status("b") == signal);
    CHECK(f.behavior("n").evaluations == 1);
  }
}

void testCopy() {
  Fixture f(R"mlir(
    func.func @f() {
      %c2 = arith.constant 2 : i32
      %two = arith.constant 2 : i32
      %n = builtin.unrealized_conversion_cast to i32 loc("n")
      %m = builtin.unrealized_conversion_cast to i32 loc("m")
      %a = scf.execute_region -> i32 {
        %y = arith.subi %c2, %n overflow<nsw> : i32
        scf.yield %y : i32
      } loc("a")
      %b = scf.execute_region -> i32 {
        %y = arith.subi %two, %m overflow<nsw> : i32
        scf.yield %y : i32
      } loc("b")
      %flags = scf.execute_region -> i32 {
        %y = arith.subi %two, %m : i32
        scf.yield %y : i32
      } loc("flags")
      return
    }
  )mlir");
  // A new 3 in the query allocator, which dies with the query.
  for (StringRef name : {"n", "m"})
    f.behavior(name).evaluate = [](Operation *, EvalScope &scope) {
      return SmallVector<Answer>{
          IntEvalValue::get(scope.getAllocator(), APInt(32, 3))};
    };

  llvm::BumpPtrAllocator cacheAllocator;
  EvalValueStorageAllocator values(f.evalContext, /*is_cache=*/true);
  CacheKey copy;
  f.inQuery([&](EvalScope &scope) {
    CacheKey source = f.key("a", scope);
    copy = copyCacheKey(source, cacheAllocator, values);
    CHECK(keysEqual(source, copy));
    CHECK(copy.op != source.op);
    CHECK(copy.inputs[1] != source.inputs[1] && copy.inputs[1].isCached());
  });

  // The copy reads neither the erased op nor the dead query's values.
  f.op("a")->erase();
  f.inQuery([&](EvalScope &scope) {
    CHECK(keysEqual(copy, f.key("b", scope)));
    CHECK(!keysEqual(copy, f.key("flags", scope)));
  });
  destroyCacheKey(copy);
}

} // namespace

int main() {
  testStructure();
  testRegions();
  testBlocks();
  testLoop();
  testGraphRegions();
  testInputs();
  testSignals();
  testCopy();
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
