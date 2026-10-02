#include "interpreter/EvalCache.h"

#include <cassert>

namespace mlir::interpreter {
namespace {

CacheKeyResult<llvm::hash_code> getHash(Operation *op, Engine &engine) {
  if (auto evaluable = dyn_cast<EvaluableOpInterface>(op))
    return evaluable.getHash(engine);
  return llvm::hash_value(op);
}

AnswerKind checkedSignal(AnswerKind kind) {
  assert((kind == AnswerKind::Exhausted || kind == AnswerKind::NeedsOrder) &&
         "cache key methods may only signal exhaustion or needs order");
  return kind;
}

}

CacheKeyResult<CacheEntry *>
EvalCache::findEquivalent(Operation *op, llvm::hash_code key, Engine &engine) {
  auto evaluable = dyn_cast<EvaluableOpInterface>(op);
  size_t checked = 0;
  while (true) {
    SmallVector<CacheEntry *> candidates;
    {
      auto it = entries.find(key);
      if (it == entries.end() || it->second.size() == checked)
        return static_cast<CacheEntry *>(nullptr);
      candidates.append(it->second.begin() + checked, it->second.end());
      checked = it->second.size();
    }
    for (CacheEntry *entry : candidates) {
      if (entry->op == op)
        return entry;
      if (!evaluable)
        continue;
      auto equal = evaluable.isEqual(entry->op, engine);
      if (auto *signal = std::get_if<AnswerKind>(&equal))
        return checkedSignal(*signal);
      if (std::get<bool>(equal)) {
        assert(entry->results.size() == op->getNumResults() &&
               "equivalent operations must have matching result counts");
        return entry;
      }
    }
  }
}

CacheResult EvalCache::lookup(Operation *op, Engine &engine) {
  auto hash = getHash(op, engine);
  if (auto *signal = std::get_if<AnswerKind>(&hash))
    return checkedSignal(*signal);
  auto key = std::get<llvm::hash_code>(hash);
  auto found = findEquivalent(op, key, engine);
  if (auto *signal = std::get_if<AnswerKind>(&found))
    return *signal;
  if (CacheEntry *entry = std::get<CacheEntry *>(found))
    return entry;
  return key;
}

CacheKeyResult<CacheEntry *>
EvalCache::insert(Operation *op, ArrayRef<Answer> answers, Engine &engine) {
  auto hash = getHash(op, engine);
  if (auto *signal = std::get_if<AnswerKind>(&hash))
    return checkedSignal(*signal);
  return insert(op, std::get<llvm::hash_code>(hash), answers, engine);
}

CacheKeyResult<CacheEntry *> EvalCache::insert(Operation *op,
                                               llvm::hash_code key,
                                               ArrayRef<Answer> answers,
                                               Engine &engine) {
  assert(answers.size() == op->getNumResults() &&
         "cache insertion requires one answer per result");
  auto found = findEquivalent(op, key, engine);
  if (std::holds_alternative<AnswerKind>(found))
    return found;
  CacheEntry *entry = std::get<CacheEntry *>(found);
  if (!entry) {
    auto owned = std::make_unique<CacheEntry>();
    owned->op = op;
    owned->results.resize(answers.size());
    entry = owned.get();
    ownedEntries.push_back(std::move(owned));
    entries[key].push_back(entry);
  }
  for (auto [slot, answer] : llvm::zip(entry->results, answers)) {
    if (slot || (answer.kind != AnswerKind::Known &&
                 answer.kind != AnswerKind::Unknown))
      continue;
    assert((answer.kind == AnswerKind::Known) == answer.value.has_value() &&
           "only known answers have values");
    Answer retained = answer;
    if (retained.value)
      retained.value = allocator.retain(*retained.value);
    slot = retained;
  }
  return entry;
}

}
