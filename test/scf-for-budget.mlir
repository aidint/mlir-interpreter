// RUN: interpreter --budget=9 %s | FileCheck %s --match-full-lines
// RUN: interpreter --budget=8 %s | FileCheck %s --match-full-lines --check-prefix=SHORT

// The loop costs a step, and each of its 4 iterations costs one for its
// binding and one for the addition, so it needs 9 steps. A loop that runs out
// is retried by the next query, since signals aren't cached.

// CHECK: value   evaluated   cache    status
// SHORT: value   evaluated   cache    status
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK:      %sum    6           miss     completed
  // SHORT:      %sum    —           miss     exhausted
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (index) {
    // CHECK-NEXT: %next   unknown     miss     completed
    // SHORT-NEXT: %next   unknown     miss     completed
    %next = arith.addi %acc, %i : index
    scf.yield %next : index
  }
  // CHECK-NEXT: %again  6           hit      completed
  // SHORT-NEXT: %again  —           miss     exhausted
  %again = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (index) {
    // CHECK-NEXT: %n      unknown     miss     completed
    // SHORT-NEXT: %n      unknown     miss     completed
    %n = arith.addi %acc, %i : index
    scf.yield %n : index
  }
  return
}
