#ifndef INTERPRETER_EVALCACHEKEY_H
#define INTERPRETER_EVALCACHEKEY_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"

#include <cassert>
#include <utility>

namespace mlir {
class Block;
class Operation;
class Region;
} // namespace mlir

namespace mlir::interpreter::detail {

/// This class represents a value used by an operation in a snapshot. An
/// `Inner` value is the `index`-th value defined inside the snapshot, in
/// pre-order. An `Input` is the `index`-th use of a value defined outside it,
/// so a value used twice takes two input slots.
struct OperandRef {
  enum Kind : unsigned { Inner, Input };
  unsigned kind : 1;
  unsigned index : 31;

  static OperandRef getInner(unsigned index) { return get(Inner, index); }
  static OperandRef getInput(unsigned index) { return get(Input, index); }

private:
  static OperandRef get(Kind kind, unsigned index) {
    assert(index < (1u << 31) && "too many values for an operand index");
    return {kind, index};
  }
};

inline bool operator==(OperandRef lhs, OperandRef rhs) {
  return lhs.kind == rhs.kind && lhs.index == rhs.index;
}
inline bool operator!=(OperandRef lhs, OperandRef rhs) { return !(lhs == rhs); }
inline llvm::hash_code hash_value(OperandRef ref) {
  return llvm::hash_combine(ref.kind, ref.index);
}

struct EvalCacheRegion;

/// This class represents the structure of an operation and its regions as
/// plain data, without any reference to the live IR except its properties.
/// Locations, SSA names and operand types are left out: an `Inner` operand gets
/// its type from its definition, and an `Input` from the input it names.
struct EvalCacheOp {
  OperationName name;
  DictionaryAttr attrs;
  /// The live op's storage, or null when the op has no properties.
  PropertyRef properties;
  ArrayRef<Type> resultTypes;
  ArrayRef<OperandRef> operands;
  /// The index of each successor in the region holding the op.
  ArrayRef<unsigned> successors;
  ArrayRef<EvalCacheRegion> regions;
};

/// This class represents a block of a snapshot.
struct EvalCacheBlock {
  ArrayRef<Type> argTypes;
  ArrayRef<EvalCacheOp> ops;
};

/// This class represents a region of a snapshot.
struct EvalCacheRegion {
  ArrayRef<EvalCacheBlock> blocks;
};

bool operator==(const EvalCacheOp &lhs, const EvalCacheOp &rhs);
bool operator==(const EvalCacheBlock &lhs, const EvalCacheBlock &rhs);
bool operator==(const EvalCacheRegion &lhs, const EvalCacheRegion &rhs);
llvm::hash_code hash_value(const EvalCacheOp &op);
llvm::hash_code hash_value(const EvalCacheBlock &block);
llvm::hash_code hash_value(const EvalCacheRegion &region);

/// This class builds the snapshot of an operation and collects the outer
/// values it uses, in input-slot order. It evaluates nothing.
///
/// Definitions are numbered in one pre-order walk: an op's results when the op
/// is visited, then its operands, then its regions, and a block's arguments
/// when the block is visited. A use can come before its definition, so an
/// operand that isn't numbered yet is parked, and only becomes an input if it
/// is still not numbered after the walk.
class CacheKeyBuilder {
public:
  explicit CacheKeyBuilder(llvm::BumpPtrAllocator &allocator)
      : allocator(allocator) {}

  /// Snapshots `root` into the allocator. The snapshot reads the properties of
  /// the live ops, so it is only valid while they are. `root` can't have
  /// successors, as they are outside its snapshot.
  const EvalCacheOp *build(Operation *root);

  /// Returns the value of each input slot of the last snapshot.
  ArrayRef<Value> getOuterValues() const { return outerValues; }

private:
  EvalCacheOp snapshotOp(Operation &op);
  EvalCacheRegion snapshotRegion(Region &region);
  EvalCacheBlock snapshotBlock(Block &block);
  ArrayRef<OperandRef> snapshotOperands(OperandRange operands);
  void number(ValueRange values);

  /// Allocates one element per element of `range`, built by `fn` in order.
  template <typename T, typename RangeT, typename FnT>
  ArrayRef<T> map(RangeT &&range, FnT fn);

  llvm::BumpPtrAllocator &allocator;
  /// The number of each value defined in the snapshot so far.
  DenseMap<Value, unsigned> inner;
  /// The index of each block in its region.
  DenseMap<Block *, unsigned> blockIndices;
  /// Operands that weren't numbered when visited, in walk order.
  SmallVector<std::pair<OperandRef *, Value>> parked;
  SmallVector<Value> outerValues;
};

} // namespace mlir::interpreter::detail

#endif
