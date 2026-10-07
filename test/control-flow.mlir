// RUN: interpreter %s --split-input-file --allow-unregistered-dialect | FileCheck %s --match-full-lines

// Every block and nested region of @main is reported in source order. Block
// arguments and ops without an evaluation function are unknown, so arguments,
// branches, regions and calls never stop a run.

// CHECK: value   evaluated   cache    status
func.func @main(%arg: i32) -> i32 {
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : i32
  // CHECK-NEXT: %a      unknown     miss     completed
  %a = arith.addi %arg, %c1 : i32
  // CHECK-NEXT: %b      2           miss     completed
  %b = arith.addi %c1, %c1 : i32
  // CHECK-NEXT: %cond   true        —        completed
  %cond = arith.constant true
  cf.cond_br %cond, ^bb1(%b : i32), ^bb2
^bb1(%x: i32):
  // CHECK-NEXT: %d      unknown     miss     completed
  %d = arith.addi %x, %c1 : i32
  // Another block shares the cache.
  // CHECK-NEXT: %e      2           hit      completed
  %e = arith.addi %c1, %c1 : i32
  cf.br ^bb2
^bb2:
  // CHECK-NEXT: %r      unknown     —        completed
  // CHECK-NEXT: %f      4           miss     completed
  %r = scf.if %cond -> i32 {
    %f = arith.muli %b, %b : i32
    scf.yield %f : i32
  } else {
    scf.yield %c1 : i32
  }
  // CHECK-NEXT: %g      unknown     miss     completed
  %g = arith.addi %r, %c1 : i32
  // CHECK-NEXT: %h      unknown     —        completed
  %h = func.call @main(%g) : (i32) -> i32
  return %h : i32
}

// -----

// A graph region can make a value depend on itself, which ends in a cycle.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : i32
  "test.graph"() ({
    // CHECK-NEXT: %a      —           —        cycle
    %a = arith.addi %b, %c1 : i32
    // CHECK-NEXT: %b      —           —        cycle
    %b = arith.addi %a, %c1 : i32
  }) : () -> ()
  return
}
