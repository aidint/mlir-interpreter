#include "interpreter/EvalCacheKey.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

#include "llvm/ADT/STLExtras.h"

#include <iterator>

namespace mlir::interpreter::detail {

bool operator==(const EvalCacheOp &lhs, const EvalCacheOp &rhs) {
  // Properties are only comparable once the names, which give their layout,
  // match.
  return lhs.name == rhs.name && lhs.attrs == rhs.attrs &&
         lhs.name.compareOpProperties(lhs.properties, rhs.properties) &&
         lhs.resultTypes == rhs.resultTypes && lhs.operands == rhs.operands &&
         lhs.successors == rhs.successors && lhs.regions == rhs.regions;
}

bool operator==(const EvalCacheBlock &lhs, const EvalCacheBlock &rhs) {
  return lhs.argTypes == rhs.argTypes && lhs.ops == rhs.ops;
}

bool operator==(const EvalCacheRegion &lhs, const EvalCacheRegion &rhs) {
  return lhs.blocks == rhs.blocks;
}

llvm::hash_code hash_value(const EvalCacheOp &op) {
  return llvm::hash_combine(
      op.name, op.attrs, op.name.hashOpProperties(op.properties),
      op.resultTypes, op.operands, op.successors, op.regions);
}

llvm::hash_code hash_value(const EvalCacheBlock &block) {
  return llvm::hash_combine(block.argTypes, block.ops);
}

llvm::hash_code hash_value(const EvalCacheRegion &region) {
  return hash_value(region.blocks);
}

template <typename T, typename RangeT, typename FnT>
ArrayRef<T> CacheKeyBuilder::map(RangeT &&range, FnT fn) {
  size_t size = std::distance(std::begin(range), std::end(range));
  if (size == 0)
    return {};
  T *data = allocator.Allocate<T>(size);
  T *next = data;
  for (auto &&element : range)
    new (next++) T(fn(element));
  return {data, size};
}

const EvalCacheOp *CacheKeyBuilder::build(Operation *root) {
  assert(root->getNumSuccessors() == 0 &&
         "a root's successors are outside its snapshot");
  inner.clear();
  blockIndices.clear();
  parked.clear();
  outerValues.clear();
  auto *snapshot =
      new (allocator.Allocate<EvalCacheOp>()) EvalCacheOp(snapshotOp(*root));
  // A parked operand is either defined later in the walk, or outside the
  // snapshot, in which case it takes the next input slot.
  for (auto [ref, value] : parked) {
    if (auto it = inner.find(value); it != inner.end()) {
      *ref = OperandRef::getInner(it->second);
      continue;
    }
    *ref = OperandRef::getInput(outerValues.size());
    outerValues.push_back(value);
  }
  return snapshot;
}

EvalCacheOp CacheKeyBuilder::snapshotOp(Operation &op) {
  // The fields are built in walk order: results, operands, then regions.
  number(op.getResults());
  ArrayRef<Type> resultTypes =
      map<Type>(op.getResultTypes(), [](Type type) { return type; });
  ArrayRef<OperandRef> operands = snapshotOperands(op.getOperands());
  ArrayRef<unsigned> successors = map<unsigned>(
      op.getSuccessors(), [&](Block *block) { return blockIndices.at(block); });
  ArrayRef<EvalCacheRegion> regions = map<EvalCacheRegion>(
      op.getRegions(), [&](Region &region) { return snapshotRegion(region); });
  return {op.getName(),
          op.getRawDictionaryAttrs(),
          op.getPropertiesStorage(),
          resultTypes,
          operands,
          successors,
          regions};
}

EvalCacheRegion CacheKeyBuilder::snapshotRegion(Region &region) {
  // A terminator can name a block that comes later, so every block is indexed
  // before any is visited.
  for (auto [index, block] : llvm::enumerate(region))
    blockIndices.try_emplace(&block, index);
  return {map<EvalCacheBlock>(
      region, [&](Block &block) { return snapshotBlock(block); })};
}

EvalCacheBlock CacheKeyBuilder::snapshotBlock(Block &block) {
  number(block.getArguments());
  ArrayRef<Type> argTypes =
      map<Type>(block.getArgumentTypes(), [](Type type) { return type; });
  ArrayRef<EvalCacheOp> ops =
      map<EvalCacheOp>(block, [&](Operation &op) { return snapshotOp(op); });
  return {argTypes, ops};
}

ArrayRef<OperandRef> CacheKeyBuilder::snapshotOperands(OperandRange operands) {
  if (operands.empty())
    return {};
  auto *refs = allocator.Allocate<OperandRef>(operands.size());
  for (auto [i, operand] : llvm::enumerate(operands)) {
    if (auto it = inner.find(operand); it != inner.end()) {
      refs[i] = OperandRef::getInner(it->second);
      continue;
    }
    // `build` overwrites this placeholder after the walk.
    refs[i] = OperandRef::getInput(0);
    parked.emplace_back(&refs[i], operand);
  }
  return {refs, operands.size()};
}

void CacheKeyBuilder::number(ValueRange values) {
  for (Value value : values)
    inner.try_emplace(value, inner.size());
}

} // namespace mlir::interpreter::detail
