#include "interpreter/EvalCache.h"

#include <cassert>
#include <memory>

namespace mlir::interpreter::detail {

EvalResult<CacheLookup> EvalCache::lookup(Operation *op, EvalScope &scope,
                                          llvm::BumpPtrAllocator &scratch) {
  auto key = buildCacheKey(op, scope, scratch);
  if (!key.completed())
    return key.status;
  CacheLookup result{key.getValue(), {}};
  if (auto it = entries.find(result.key); it != entries.end())
    result.results = it->second;
  return result;
}

MutableArrayRef<std::optional<Answer>>
EvalCache::insert(const CacheKey &key, ArrayRef<Answer> answers) {
  // Evaluating the op can run nested queries that add an entry for an equal
  // key, so look it up again instead of trusting the lookup's miss.
  auto it = entries.find(key);
  if (it == entries.end()) {
    auto *results =
        entryAllocator.Allocate<std::optional<Answer>>(answers.size());
    std::uninitialized_fill_n(results, answers.size(), std::nullopt);
    CacheKey copy = copyCacheKey(key, entryAllocator, allocator);
    it = entries.try_emplace(copy, results, answers.size()).first;
  }
  fill(it->second, answers);
  return it->second;
}

void EvalCache::fill(MutableArrayRef<std::optional<Answer>> results,
                     ArrayRef<Answer> answers) {
  assert(answers.size() == results.size() &&
         "cache filling requires one answer per result");
  for (auto [slot, answer] : llvm::zip(results, answers)) {
    assert((answer.completed() || !answer.value) &&
           "incomplete evaluations cannot have a value");
    // TODO: Store unknowns too, now that keys include the values a loop binds
    // (MLIR-36).
    if (slot || !answer.completed() || answer.isUnknown())
      continue;
    slot = allocator.retain(answer.getValue());
  }
}

void EvalCache::destroyKeys() {
  for (const auto &entry : entries)
    destroyCacheKey(entry.first);
}

} // namespace mlir::interpreter::detail
