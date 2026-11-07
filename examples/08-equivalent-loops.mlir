// RUN: interpreter %s | FileCheck %s --match-full-lines

// A loop is keyed by its evaluated operands and its body's structure, so a
// loop over other SSA values with equal values hits.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c0     0           —        completed
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %c2     2           —        completed
  %c2 = arith.constant 2 : index
  // CHECK-NEXT: %c4     4           —        completed
  %c4 = arith.constant 4 : index
  // CHECK-NEXT: %a      6           miss     completed
  %a = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (index) {
    // CHECK-NEXT: %next   unknown     miss     completed
    %next = arith.addi %acc, %i : index
    scf.yield %next : index
  }
  // CHECK-NEXT: %zero   0           miss     completed
  %zero = arith.subi %c2, %c2 : index
  // CHECK-NEXT: %four   4           miss     completed
  %four = arith.muli %c2, %c2 : index
  // Other bounds and initial value, and other block argument names, but the
  // same values and body.
  // CHECK-NEXT: %b      6           hit      completed
  %b = scf.for %j = %zero to %four step %c1 iter_args(%s = %zero) -> (index) {
    // CHECK-NEXT: %t      unknown     miss     completed
    %t = arith.addi %s, %j : index
    scf.yield %t : index
  }
  // Another body misses: 0 * 1 * 2 * 3.
  // CHECK-NEXT: %c      0           miss     completed
  %c = scf.for %j = %zero to %four step %c1 iter_args(%s = %zero) -> (index) {
    // CHECK-NEXT: %u      unknown     miss     completed
    %u = arith.muli %s, %j : index
    scf.yield %u : index
  }
  return
}
