// RUN: interpreter --budget=3 %s | FileCheck %s --match-full-lines

// Each reported value gets 3 steps. Constants are not cached, so a query pays
// a step for each constant it reaches, even to compute the key of a hit.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : i32
  // CHECK-NEXT: %c2     2           —        completed
  %c2 = arith.constant 2 : i32
  // CHECK-NEXT: %c3     3           —        completed
  %c3 = arith.constant 3 : i32
  // CHECK-NEXT: %c4     4           —        completed
  %c4 = arith.constant 4 : i32
  // arith.divsi has no evaluation function, so %u is unknown and not cached.
  // CHECK-NEXT: %u      unknown     —        completed
  %u = arith.divsi %c4, %c2 : i32
  // An unknown left operand keys the sum by its operation...
  // CHECK-NEXT: %a      unknown     miss     completed
  %a = arith.addi %u, %c1 : i32
  // ...so an identical sum gets its own entry.
  // CHECK-NEXT: %b      unknown     miss     completed
  %b = arith.addi %u, %c1 : i32
  // 2 steps for the constants and 1 for the sum.
  // CHECK-NEXT: %s      3           miss     completed
  %s = arith.addi %c1, %c2 : i32
  // The key costs 2 steps, and a hit needs no more.
  // CHECK-NEXT: %h      3           hit      completed
  %h = arith.addi %c2, %c1 : i32
  // The key reaches 3 constants, so the lookup completes but no step
  // is left for the sum.
  // CHECK-NEXT: %e      —           miss     exhausted
  %e = arith.addi %s, %c3 : i32
  // 2 steps for the constants and 1 for the sum.
  // CHECK-NEXT: %t      7           miss     completed
  %t = arith.addi %c3, %c4 : i32
  // The key would reach 4 constants, so hashing runs out before a lookup.
  // CHECK-NEXT: %f      —           —        exhausted
  %f = arith.addi %s, %t : i32
  return
}
