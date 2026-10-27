// RUN: interpreter %s | FileCheck %s --match-full-lines

// A loop binds its induction variable and iteration arguments for each
// iteration. Rows inside a loop body are queried outside any iteration, so the
// body's block arguments are unknown there.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c0     0           —        completed
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %c3     3           —        completed
  %c3 = arith.constant 3 : index
  // CHECK-NEXT: %c4     4           —        completed
  %c4 = arith.constant 4 : index
  // 0 + 1 + 2 + 3
  // CHECK-NEXT: %sum    6           miss     completed
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (index) {
    // CHECK-NEXT: %next   unknown     miss     completed
    %next = arith.addi %acc, %i : index
    scf.yield %next : index
  }
  // The inner loop runs again in each outer iteration, so %i stays bound
  // while %j iterates: the sum of i * j for i < 3 and j < 4.
  // CHECK-NEXT: %prod   18          miss     completed
  %prod = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %c0) -> (index) {
    // CHECK-NEXT: %inner  unknown     miss     completed
    %inner = scf.for %j = %c0 to %c4 step %c1 iter_args(%a = %acc) -> (index) {
      // CHECK-NEXT: %ij     unknown     miss     completed
      %ij = arith.muli %i, %j : index
      // CHECK-NEXT: %n      unknown     miss     completed
      %n = arith.addi %a, %ij : index
      scf.yield %n : index
    }
    scf.yield %inner : index
  }
  return
}
