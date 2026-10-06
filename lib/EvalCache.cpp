#include "interpreter/EvalCache.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include <cassert>
#include <memory>

namespace mlir::interpreter {
namespace {

EvalStatus checkedSignal(EvalStatus status) {
  assert(
      (status == EvalStatus::Exhausted || status == EvalStatus::NeedsOrder) &&
      "cache key methods may only signal exhaustion or needs order");
  return status;
}

}

EvalResult<CacheEntry *>
EvalCache::findEquivalent(EvaluableOpInterface op, llvm::hash_code key,
                          Engine &engine) {
  size_t checked = 0;
  while (true) {
    SmallVector<CacheEntry *> candidates;
    {
      auto it = entries.find(key);
      if (it == entries.end() || it->second.size() == checked)
        return {EvalStatus::Completed, std::nullopt};
      candidates.append(it->second.begin() + checked, it->second.end());
      checked = it->second.size();
    }
    for (CacheEntry *entry : candidates) {
      if (entry->op == op)
        return {EvalStatus::Completed, entry};
      auto equal = op.isEqual(entry->op, engine);
      const auto &isEqual = equal.getValue();
      if (equal.status != EvalStatus::Completed)
        return {checkedSignal(equal.status), std::nullopt};
      assert(isEqual && "completed equality must provide a result");
      if (*isEqual) {
        assert(entry->results.size() == op->getNumResults() &&
               "equivalent operations must have matching result counts");
        return {EvalStatus::Completed, entry};
      }
    }
  }
}

EvalResult<CacheLookup> EvalCache::lookup(Operation *op, Engine &engine) {
  auto evaluable = cast<EvaluableOpInterface>(op);
  auto hash = evaluable.getHash(engine);
  const auto &hashValue = hash.getValue();
  if (hash.status != EvalStatus::Completed)
    return {checkedSignal(hash.status), std::nullopt};
  assert(hashValue && "completed hashing must provide a hash");
  auto key = *hashValue;
  auto found = findEquivalent(evaluable, key, engine);
  if (found.status != EvalStatus::Completed)
    return {found.status, std::nullopt};
  return {EvalStatus::Completed,
          CacheLookup{key, found.getValue().value_or(nullptr)}};
}

EvalResult<CacheEntry *>
EvalCache::insert(Operation *op, ArrayRef<Answer> answers, Engine &engine) {
  auto hash = cast<EvaluableOpInterface>(op).getHash(engine);
  const auto &hashValue = hash.getValue();
  if (hash.status != EvalStatus::Completed)
    return {checkedSignal(hash.status), std::nullopt};
  assert(hashValue && "completed hashing must provide a hash");
  return insert(op, *hashValue, answers, engine);
}

EvalResult<CacheEntry *> EvalCache::insert(Operation *op, llvm::hash_code key,
                                           ArrayRef<Answer> answers,
                                           Engine &engine) {
  assert(answers.size() == op->getNumResults() &&
         "cache insertion requires one answer per result");
  auto found = findEquivalent(cast<EvaluableOpInterface>(op), key, engine);
  if (found.status != EvalStatus::Completed)
    return found;
  CacheEntry *entry = found.getValue().value_or(nullptr);
  if (!entry) {
    auto *results =
        entryAllocator.Allocate<std::optional<Answer>>(answers.size());
    std::uninitialized_fill_n(results, answers.size(), std::nullopt);
    entry = new (entryAllocator.Allocate<CacheEntry>())
        CacheEntry{op, {results, answers.size()}};
    entries[key].push_back(entry);
  }
  fill(entry, answers);
  return {EvalStatus::Completed, entry};
}

void EvalCache::fill(CacheEntry *entry, ArrayRef<Answer> answers) {
  assert(answers.size() == entry->results.size() &&
         "cache filling requires one answer per result");
  for (auto [slot, answer] : llvm::zip(entry->results, answers)) {
    assert((answer.status == EvalStatus::Completed || !answer.value) &&
           "incomplete evaluations cannot have a value");
    if (slot || answer.status != EvalStatus::Completed)
      continue;
    Answer retained = answer;
    if (const auto &value = answer.getValue())
      retained.value = allocator.retain(*value);
    slot = retained;
  }
}

}
