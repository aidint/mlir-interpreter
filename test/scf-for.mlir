// RUN: interpreter %s --split-input-file | FileCheck %s --match-full-lines

// Nested loops. An op in a running loop skips the cache, so an inner loop is
// evaluated again in each outer iteration, even when its operands are equal.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c0     0           —        completed
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %c4     4           —        completed
  %c4 = arith.constant 4 : index
  // The inner bound is the outer induction variable: 0 + 1 + 2 + 3.
  // CHECK-NEXT: %tri    6           miss     completed
  %tri = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (index) {
    // CHECK-NEXT: %inner  unknown     miss     completed
    %inner = scf.for %j = %c0 to %i step %c1 iter_args(%a = %acc) -> (index) {
      // CHECK-NEXT: %n      unknown     miss     completed
      %n = arith.addi %a, %c1 : index
      scf.yield %n : index
    }
    scf.yield %inner : index
  }
  // The inner loop's operands are equal in every outer iteration, but its body
  // captures %i, so a cached answer would be stale: 0 + 4 + 8 + 12.
  // CHECK-NEXT: %cap    24          miss     completed
  %cap = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (index) {
    // Outside an iteration, %i is unknown, so %row is too.
    // CHECK-NEXT: %row    unknown     miss     completed
    %row = scf.for %j = %c0 to %c4 step %c1 iter_args(%a = %c0) -> (index) {
      // CHECK-NEXT: %m      unknown     miss     completed
      %m = arith.addi %a, %i : index
      scf.yield %m : index
    }
    // CHECK-NEXT: %next   unknown     miss     completed
    %next = arith.addi %acc, %row : index
    scf.yield %next : index
  }
  return
}

// -----

// Iteration arguments are bound together, so a yield can swap them. Both
// results are computed at once, so the second one hits.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c0     0           —        completed
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %c10    10          —        completed
  %c10 = arith.constant 10 : index
  // CHECK-NEXT: %f0     0           —        completed
  %f0 = arith.constant 0 : i32
  // CHECK-NEXT: %f1     1           —        completed
  %f1 = arith.constant 1 : i32
  // CHECK-NEXT: %fib#0  55          miss     completed
  // CHECK-NEXT: %fib#1  89          hit      completed
  %fib:2 = scf.for %i = %c0 to %c10 step %c1
      iter_args(%a = %f0, %b = %f1) -> (i32, i32) {
    // CHECK-NEXT: %s      unknown     miss     completed
    %s = arith.addi %a, %b : i32
    scf.yield %b, %s : i32, i32
  }
  return
}

// -----

// Bounds, steps and initial values the loop can't run with.

// CHECK: value   evaluated   cache    status
func.func @main(%arg: index) {
  // CHECK-NEXT: %c0     0           —        completed
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %c4     4           —        completed
  %c4 = arith.constant 4 : index
  // A loop that doesn't run yields its initial value.
  // CHECK-NEXT: %none   4           miss     completed
  %none = scf.for %i = %c4 to %c0 step %c1 iter_args(%a = %c4) -> (index) {
    scf.yield %c0 : index
  }
  // CHECK-NEXT: %ub     unknown     miss     completed
  %ub = scf.for %i = %c0 to %arg step %c1 iter_args(%a = %c0) -> (index) {
    scf.yield %i : index
  }
  // CHECK-NEXT: %init   unknown     miss     completed
  %init = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %arg) -> (index) {
    scf.yield %i : index
  }
  // A step that isn't strictly positive is undefined.
  // CHECK-NEXT: %neg    -1          miss     completed
  %neg = arith.subi %c0, %c1 : index
  // CHECK-NEXT: %back   unknown     miss     completed
  %back = scf.for %i = %c4 to %c0 step %neg iter_args(%a = %c0) -> (index) {
    scf.yield %i : index
  }
  // CHECK-NEXT: %b0     0           —        completed
  %b0 = arith.constant 0 : i8
  // CHECK-NEXT: %b1     1           —        completed
  %b1 = arith.constant 1 : i8
  // CHECK-NEXT: %b50    50          —        completed
  %b50 = arith.constant 50 : i8
  // CHECK-NEXT: %b100   100         —        completed
  %b100 = arith.constant 100 : i8
  // CHECK-NEXT: %b127   127         —        completed
  %b127 = arith.constant 127 : i8
  // CHECK-NEXT: %b200   -56         —        completed
  %b200 = arith.constant 200 : i8
  // Past 0 and 100, the induction variable would overflow, which is undefined.
  // CHECK-NEXT: %over   unknown     miss     completed
  %over = scf.for %i = %b0 to %b127 step %b100 iter_args(%a = %b0) -> (i8) : i8 {
    scf.yield %i : i8
  }
  // Signed, 200 is -56, so the loop doesn't run. Unsigned, it runs for 0, 50,
  // 100 and 150.
  // CHECK-NEXT: %slt    0           miss     completed
  %slt = scf.for %i = %b0 to %b200 step %b50 iter_args(%a = %b0) -> (i8) : i8 {
    // CHECK-NEXT: %s      unknown     miss     completed
    %s = arith.addi %a, %b1 : i8
    scf.yield %s : i8
  }
  // CHECK-NEXT: %ult    4           miss     completed
  %ult = scf.for unsigned %i = %b0 to %b200 step %b50
      iter_args(%a = %b0) -> (i8) : i8 {
    // CHECK-NEXT: %u      unknown     miss     completed
    %u = arith.addi %a, %b1 : i8
    scf.yield %u : i8
  }
  return
}

// -----

// Each iteration costs a step, so a loop whose body costs nothing still runs
// out of budget.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c0     0           —        completed
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : index
  // CHECK-NEXT: %big    1000000000  —        completed
  %big = arith.constant 1000000000 : index
  // CHECK-NEXT: %free   —           miss     exhausted
  %free = scf.for %i = %c0 to %big step %c1 iter_args(%a = %c0) -> (index) {
    scf.yield %a : index
  }
  return
}
