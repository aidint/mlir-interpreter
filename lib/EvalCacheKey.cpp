#include "interpreter/EvalCacheKey.h"
#include "interpreter/Answer.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Alignment.h"

#include <iterator>

namespace mlir::interpreter::detail {
namespace {

/// Allocates one element per element of `range` in `allocator`, built by `fn`
/// in order.
template <typename T, typename RangeT, typename FnT>
ArrayRef<T> allocateArray(llvm::BumpPtrAllocator &allocator, RangeT &&range,
                          FnT fn) {
  size_t size = std::distance(std::begin(range), std::end(range));
  if (size == 0)
    return {};
  T *data = allocator.Allocate<T>(size);
  T *next = data;
  for (auto &&element : range)
    new (next++) T(fn(element));
  return {data, size};
}

template <typename T>
ArrayRef<T> copyArray(llvm::BumpPtrAllocator &allocator, ArrayRef<T> array) {
  return allocateArray<T>(allocator, array,
                          [](const T &element) { return element; });
}

EvalCacheOp copyOp(const EvalCacheOp &op, llvm::BumpPtrAllocator &allocator);

EvalCacheBlock copyBlock(const EvalCacheBlock &block,
                         llvm::BumpPtrAllocator &allocator) {
  ArrayRef<EvalCacheOp> ops = allocateArray<EvalCacheOp>(
      allocator, block.ops,
      [&](const EvalCacheOp &op) { return copyOp(op, allocator); });
  return {copyArray(allocator, block.argTypes), ops};
}

EvalCacheRegion copyRegion(const EvalCacheRegion &region,
                           llvm::BumpPtrAllocator &allocator) {
  return {allocateArray<EvalCacheBlock>(allocator, region.blocks,
                                        [&](const EvalCacheBlock &block) {
                                          return copyBlock(block, allocator);
                                        })};
}

EvalCacheOp copyOp(const EvalCacheOp &op, llvm::BumpPtrAllocator &allocator) {
  PropertyRef properties;
  if (op.properties) {
    // Properties are aligned to 8 bytes, as in `Operation`.
    void *storage =
        allocator.Allocate(op.name.getOpPropertyByteSize(), llvm::Align(8));
    properties = PropertyRef(op.properties.getTypeID(), storage);
    op.name.initOpProperties(properties, op.properties);
  }
  ArrayRef<EvalCacheRegion> regions = allocateArray<EvalCacheRegion>(
      allocator, op.regions, [&](const EvalCacheRegion &region) {
        return copyRegion(region, allocator);
      });
  return {op.name,
          op.attrs,
          properties,
          copyArray(allocator, op.resultTypes),
          copyArray(allocator, op.operands),
          copyArray(allocator, op.successors),
          regions};
}

void destroyOp(const EvalCacheOp &op) {
  if (op.properties)
    op.name.destroyOpProperties(op.properties);
  for (const EvalCacheRegion &region : op.regions)
    for (const EvalCacheBlock &block : region.blocks)
      for (const EvalCacheOp &nested : block.ops)
        destroyOp(nested);
}

/// Returns whether two inputs are equal. Unknown matches only unknown, and
/// `EvalValue::isEqual` can't take a null value.
bool isInputEqual(EvalValue lhs, EvalValue rhs) {
  if (!lhs || !rhs)
    return !lhs && !rhs;
  return lhs.isEqual(rhs);
}

} // namespace

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
  ArrayRef<Type> resultTypes = allocateArray<Type>(
      allocator, op.getResultTypes(), [](Type type) { return type; });
  ArrayRef<OperandRef> operands = snapshotOperands(op.getOperands());
  ArrayRef<unsigned> successors =
      allocateArray<unsigned>(allocator, op.getSuccessors(), [&](Block *block) {
        return blockIndices.at(block);
      });
  ArrayRef<EvalCacheRegion> regions = allocateArray<EvalCacheRegion>(
      allocator, op.getRegions(),
      [&](Region &region) { return snapshotRegion(region); });
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
  return {allocateArray<EvalCacheBlock>(
      allocator, region, [&](Block &block) { return snapshotBlock(block); })};
}

EvalCacheBlock CacheKeyBuilder::snapshotBlock(Block &block) {
  number(block.getArguments());
  ArrayRef<Type> argTypes = allocateArray<Type>(
      allocator, block.getArgumentTypes(), [](Type type) { return type; });
  ArrayRef<EvalCacheOp> ops = allocateArray<EvalCacheOp>(
      allocator, block, [&](Operation &op) { return snapshotOp(op); });
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

EvalResult<CacheKey> buildCacheKey(Operation *op, EvalScope &scope,
                                   llvm::BumpPtrAllocator &scratch) {
  CacheKeyBuilder builder(scratch);
  const EvalCacheOp *snapshot = builder.build(op);
  ArrayRef<Value> outerValues = builder.getOuterValues();
  ArrayRef<Type> inputTypes = allocateArray<Type>(
      scratch, outerValues, [](Value value) { return value.getType(); });
  auto *inputs = scratch.Allocate<EvalValue>(outerValues.size());
  for (auto [i, value] : llvm::enumerate(outerValues)) {
    Answer answer = scope.query(value);
    if (!answer.completed())
      return answer.status;
    new (inputs + i) EvalValue(answer.value);
  }
  return CacheKey{snapshot, inputTypes, {inputs, outerValues.size()}};
}

CacheKey copyCacheKey(const CacheKey &key, llvm::BumpPtrAllocator &allocator,
                      EvalValueStorageAllocator &values) {
  auto *op = new (allocator.Allocate<EvalCacheOp>())
      EvalCacheOp(copyOp(*key.op, allocator));
  ArrayRef<EvalValue> inputs =
      allocateArray<EvalValue>(allocator, key.inputs, [&](EvalValue input) {
        return input ? values.retain(input) : EvalValue();
      });
  return {op, copyArray(allocator, key.inputTypes), inputs};
}

void destroyCacheKey(const CacheKey &key) { destroyOp(*key.op); }

} // namespace mlir::interpreter::detail

using namespace mlir::interpreter;
using namespace mlir::interpreter::detail;

unsigned llvm::DenseMapInfo<CacheKey>::getHashValue(const CacheKey &key) {
  llvm::hash_code hash = llvm::hash_combine(*key.op, key.inputTypes);
  for (EvalValue input : key.inputs)
    hash = llvm::hash_combine(hash, input ? input.hash() : llvm::hash_code(0));
  return hash;
}

bool llvm::DenseMapInfo<CacheKey>::isEqual(const CacheKey &lhs,
                                           const CacheKey &rhs) {
  // Equal input types come first, so inputs are only compared with values of
  // their own type.
  return *lhs.op == *rhs.op && lhs.inputTypes == rhs.inputTypes &&
         llvm::all_of_zip(lhs.inputs, rhs.inputs, isInputEqual);
}
