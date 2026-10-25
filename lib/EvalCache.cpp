#include "interpreter/EvalCache.h"
#include "interpreter/Interfaces/EvaluableOpInterface.h"

#include <cassert>
#include <memory>

namespace mlir::interpreter::detail {

CacheEntry *EvalCache::findEquivalent(EvaluableOpInterface op,
                                      llvm::hash_code key, size_t &next,
                                      EvalScope &scope) {
  for (;; ++next) {
    // `isEquivalentTo` can run nested queries that insert into the map, so
    // look the bucket up again on every step. Entries are only appended and
    // their addresses are stable, so `next` stays valid.
    auto it = entries.find(key);
    if (it == entries.end() || next == it->second.size())
      return nullptr;
    CacheEntry *entry = it->second[next];
    if (entry->op == op)
      return entry;
    if (entry->results.size() != op->getNumResults())
      continue;
    // A signal only means this candidate can't be compared now; another
    // candidate, or evaluation, may still answer.
    auto equivalent = op.isEquivalentTo(entry->op, scope);
    if (!equivalent.completed())
      continue;
    if (equivalent.getValue())
      return entry;
  }
}

EvalResult<CacheLookup> EvalCache::lookup(Operation *op, EvalScope &scope) {
  auto evaluable = cast<EvaluableOpInterface>(op);
  auto hash = evaluable.getHash(scope);
  if (!hash.completed())
    return hash.status;
  CacheLookup result{hash.getValue(), nullptr, 0};
  result.entry = findEquivalent(evaluable, result.key, result.searched, scope);
  return result;
}

CacheEntry *EvalCache::insert(Operation *op, const CacheLookup &lookup,
                              ArrayRef<Answer> answers, EvalScope &scope) {
  assert(answers.size() == op->getNumResults() &&
         "cache insertion requires one answer per result");
  // Evaluating `op` can run nested queries that cache an operation equivalent
  // to it, so search again instead of adding a duplicate entry. `lookup`
  // already compared the first `searched` candidates, and keys are stable, so
  // only candidates appended since then can match. Comparing the others again
  // would spend budget on `isEquivalentTo` calls whose answer is already known.
  size_t next = lookup.searched;
  CacheEntry *entry =
      findEquivalent(cast<EvaluableOpInterface>(op), lookup.key, next, scope);
  if (!entry) {
    auto *results =
        entryAllocator.Allocate<std::optional<Answer>>(answers.size());
    std::uninitialized_fill_n(results, answers.size(), std::nullopt);
    entry = new (entryAllocator.Allocate<CacheEntry>())
        CacheEntry{op, {results, answers.size()}};
    entries[lookup.key].push_back(entry);
  }
  fill(entry, answers);
  return entry;
}

void EvalCache::fill(CacheEntry *entry, ArrayRef<Answer> answers) {
  assert(answers.size() == entry->results.size() &&
         "cache filling requires one answer per result");
  for (auto [slot, answer] : llvm::zip(entry->results, answers)) {
    assert((answer.completed() || !answer.value) &&
           "incomplete evaluations cannot have a value");
    if (slot || !answer.completed())
      continue;
    Answer retained = answer;
    if (const auto &value = answer.getValue())
      retained.value = allocator.retain(*value);
    slot = retained;
  }
}

} // namespace mlir::interpreter::detail
