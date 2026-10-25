#include "interpreter/EvalOperationEquivalence.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::interpreter {
namespace {

bool isCommutative(Operation *op) {
  return op->hasTrait<OpTrait::IsCommutative>();
}

/// This class compares two operations and their regions in three passes. The
/// first matches their structure and records which blocks, block arguments and
/// results correspond. The second checks that operands defined inside the
/// compared operations correspond. Only then does the third pass call
/// `checkEquivalent` on the remaining operands, so a structural mismatch is
/// found before any operand is evaluated.
class EquivalenceChecker {
public:
  EquivalenceChecker(
      EvalOperationEquivalence::CheckEquivalentFn checkEquivalent)
      : checkEquivalent(checkEquivalent) {}

  EvalResult<bool> check(Operation *lhs, Operation *rhs) {
    EvalResult<bool> different = false;
    if (!matchOperation(lhs, rhs))
      return different;
    // A use can precede its definition, e.g. in a block listed before the
    // block that dominates it, so operands are only checked once every value
    // is mapped.
    for (auto [l, r] : ops) {
      if (isCommutative(l))
        continue;
      for (auto [lhsOperand, rhsOperand] :
           llvm::zip_equal(l->getOperands(), r->getOperands())) {
        std::optional<bool> matched = matchValues(lhsOperand, rhsOperand);
        if (matched && !*matched)
          return different;
      }
    }
    for (auto [l, r] : ops) {
      auto equivalent =
          isCommutative(l) ? compareCommutative(l, r) : compareOperands(l, r);
      if (!equivalent.completed() || !equivalent.getValue())
        return equivalent;
    }
    return true;
  }

private:
  bool matchOperation(Operation *lhs, Operation *rhs) {
    if (lhs->getName() != rhs->getName() ||
        lhs->getNumRegions() != rhs->getNumRegions() ||
        lhs->getNumSuccessors() != rhs->getNumSuccessors() ||
        lhs->getNumOperands() != rhs->getNumOperands() ||
        lhs->getNumResults() != rhs->getNumResults() ||
        lhs->getResultTypes() != rhs->getResultTypes() ||
        lhs->getRawDictionaryAttrs() != rhs->getRawDictionaryAttrs() ||
        !lhs->getName().compareOpProperties(lhs->getPropertiesStorage(),
                                            rhs->getPropertiesStorage()))
      return false;
    // A successor outside the compared regions must be the same block.
    for (auto [l, r] :
         llvm::zip_equal(lhs->getSuccessors(), rhs->getSuccessors()))
      if (blocks.lookup_or(l, l) != r)
        return false;
    mapValues(lhs->getResults(), rhs->getResults());
    ops.emplace_back(lhs, rhs);
    return llvm::all_of_zip(
        lhs->getRegions(), rhs->getRegions(),
        [&](Region &l, Region &r) { return matchRegion(l, r); });
  }

  bool matchRegion(Region &lhs, Region &rhs) {
    if (lhs.getBlocks().size() != rhs.getBlocks().size())
      return false;
    // Blocks correspond by position. Mapping them all first lets a terminator
    // name a block that comes later.
    for (auto [l, r] : llvm::zip_equal(lhs, rhs))
      blocks.try_emplace(&l, &r);
    return llvm::all_of_zip(
        lhs, rhs, [&](Block &l, Block &r) { return matchBlock(l, r); });
  }

  bool matchBlock(Block &lhs, Block &rhs) {
    if (lhs.getNumArguments() != rhs.getNumArguments())
      return false;
    for (auto [l, r] : llvm::zip_equal(lhs.getArguments(), rhs.getArguments()))
      if (l.getType() != r.getType())
        return false;
    mapValues(lhs.getArguments(), rhs.getArguments());
    return llvm::all_of_zip(lhs, rhs, [&](Operation &l, Operation &r) {
      return matchOperation(&l, &r);
    });
  }

  void mapValues(ValueRange lhs, ValueRange rhs) {
    for (auto [l, r] : llvm::zip_equal(lhs, rhs)) {
      values.try_emplace(l, r);
      rhsValues.insert(r);
    }
  }

  /// Returns whether `lhs` and `rhs` correspond, or nothing when both are
  /// distinct values from outside the compared operations, which only
  /// `checkEquivalent` can compare.
  std::optional<bool> matchValues(Value lhs, Value rhs) const {
    if (Value mapped = values.lookup(lhs))
      return mapped == rhs;
    if (rhsValues.contains(rhs))
      return false;
    if (lhs == rhs)
      return true;
    if (lhs.getType() != rhs.getType())
      return false;
    return std::nullopt;
  }

  EvalResult<bool> compareValues(Value lhs, Value rhs) {
    if (std::optional<bool> matched = matchValues(lhs, rhs))
      return *matched;
    return checkEquivalent(lhs, rhs);
  }

  EvalResult<bool> compareOperands(Operation *lhs, Operation *rhs) {
    for (auto [l, r] :
         llvm::zip_equal(lhs->getOperands(), rhs->getOperands())) {
      auto equivalent = compareValues(l, r);
      if (!equivalent.completed() || !equivalent.getValue())
        return equivalent;
    }
    return true;
  }

  /// Matches operands as multisets: each `lhs` operand takes the first unused
  /// `rhs` operand equivalent to it, so duplicates pair up one to one.
  /// Equivalence is transitive, so this greedy matching fails only when no
  /// matching exists.
  EvalResult<bool> compareCommutative(Operation *lhs, Operation *rhs) {
    SmallVector<bool> used(rhs->getNumOperands(), false);
    for (Value l : lhs->getOperands()) {
      bool found = false;
      for (auto [r, isUsed] : llvm::zip_equal(rhs->getOperands(), used)) {
        if (isUsed)
          continue;
        auto equivalent = compareValues(l, r);
        if (!equivalent.completed())
          return equivalent;
        if (equivalent.getValue()) {
          isUsed = found = true;
          break;
        }
      }
      if (!found)
        return false;
    }
    return true;
  }

  EvalOperationEquivalence::CheckEquivalentFn checkEquivalent;
  /// Corresponding operations, in the order their operands are compared.
  SmallVector<std::pair<Operation *, Operation *>> ops;
  /// Maps blocks, block arguments and results inside `lhs` to their
  /// counterparts inside `rhs`.
  DenseMap<Block *, Block *> blocks;
  DenseMap<Value, Value> values;
  /// The values `values` maps to, so a value from outside `lhs` never matches
  /// one defined inside `rhs`.
  DenseSet<Value> rhsValues;
};

} // namespace

EvalResult<llvm::hash_code>
EvalOperationEquivalence::computeHash(Operation *op,
                                      HashOperandFn hashOperand) {
  llvm::hash_code hash =
      llvm::hash_combine(op->getName(), op->getRawDictionaryAttrs(),
                         op->getResultTypes(), op->hashProperties());
  bool commutative = isCommutative(op);
  size_t operandSum = 0;
  for (Value operand : op->getOperands()) {
    auto operandHash = hashOperand(operand);
    if (!operandHash.completed())
      return operandHash.status;
    if (commutative)
      operandSum += operandHash.getValue();
    else
      hash = llvm::hash_combine(hash, operandHash.getValue());
  }
  if (commutative)
    hash = llvm::hash_combine(hash, operandSum);
  return hash;
}

EvalResult<bool>
EvalOperationEquivalence::isEquivalentTo(Operation *lhs, Operation *rhs,
                                         CheckEquivalentFn checkEquivalent) {
  if (lhs == rhs)
    return true;
  return EquivalenceChecker(checkEquivalent).check(lhs, rhs);
}

} // namespace mlir::interpreter
