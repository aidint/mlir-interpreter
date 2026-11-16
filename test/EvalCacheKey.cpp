#include "interpreter/EvalCacheKey.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <initializer_list>

using namespace mlir;
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

/// This class parses a module whose operations are named by `NameLoc`s, and
/// snapshots them. Locations aren't part of a snapshot, so names don't change
/// it.
struct Fixture {
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  llvm::BumpPtrAllocator allocator;

  /// Graph regions of unregistered ops are only accepted without verification.
  explicit Fixture(StringRef source, bool verify = true) {
    context.loadDialect<arith::ArithDialect, cf::ControlFlowDialect,
                        func::FuncDialect, scf::SCFDialect>();
    context.allowUnregisteredDialects();
    module =
        parseSourceString<ModuleOp>(source, ParserConfig(&context, verify));
    if (!module) {
      llvm::errs() << "failed to parse the test module\n";
      std::exit(1);
    }
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

} // namespace

int main() {
  testStructure();
  testRegions();
  testBlocks();
  testLoop();
  testGraphRegions();
  if (failures)
    llvm::errs() << failures << " check(s) failed\n";
  return failures ? 1 : 0;
}
