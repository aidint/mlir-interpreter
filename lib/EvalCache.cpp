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
  if (auto it = results.find(result.key); it != results.end())
    result.results = it->second;
  return result;
}

MutableArrayRef<std::optional<Answer>>
EvalCache::insert(const CacheKey &key, ArrayRef<Answer> answers) {
  // Evaluating the op can run nested queries that add an entry for an equal
  // key, so look it up again instead of trusting the lookup's miss.
  auto it = results.find(key);
  if (it == results.end()) {
    auto *slots = slotAllocator.Allocate<std::optional<Answer>>(answers.size());
    std::uninitialized_fill_n(slots, answers.size(), std::nullopt);
    CacheKey copy = copyCacheKey(key, keyAllocator, valueAllocator);
    it = results.try_emplace(copy, slots, answers.size()).first;
  }
  fill(it->second, answers);
  return it->second;
}

void EvalCache::fill(MutableArrayRef<std::optional<Answer>> slots,
                     ArrayRef<Answer> answers) {
  assert(answers.size() == slots.size() &&
         "cache filling requires one answer per result");
  for (auto [slot, answer] : llvm::zip(slots, answers)) {
    assert((answer.completed() || !answer.value) &&
           "incomplete evaluations cannot have a value");
    // A signal describes one run, so its slot stays empty for a later query to
    // fill.
    if (slot || !answer.completed())
      continue;
    // An unknown has no value to retain.
    slot = answer.isUnknown()
               ? answer
               : Answer(valueAllocator.retain(answer.getValue()));
  }
}

void EvalCache::destroyKeys() {
  for (const auto &entry : results)
    destroyCacheKey(entry.first);
}

} // namespace mlir::interpreter::detail
