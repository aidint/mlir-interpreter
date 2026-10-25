// RUN: interpreter --budget=1 %s | FileCheck %s --match-full-lines
// RUN: interpreter --budget=0 %s | FileCheck %s --match-full-lines --check-prefix=NONE

// Constants and cache hits cost no step. With 1 step per reported value every
// operation fits, since the values it reaches are constants or already cached.
// With none, only values that need no evaluation complete.

// CHECK: value   evaluated   cache    status
// NONE:  value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c1     1           —        completed
  // NONE-NEXT:  %c1     1           —        completed
  %c1 = arith.constant 1 : i32
  // CHECK-NEXT: %c2     2           —        completed
  %c2 = arith.constant 2 : i32
  // CHECK-NEXT: %c3     3           —        completed
  %c3 = arith.constant 3 : i32
  // CHECK-NEXT: %c4     4           —        completed
  %c4 = arith.constant 4 : i32
  // arith.divsi has no evaluation function, so %u is unknown and not cached.
  // CHECK-NEXT: %u      unknown     —        completed
  // NONE:       %u      unknown     —        completed
  %u = arith.divsi %c4, %c2 : i32
  // An unknown left operand keys the sum by its operation...
  // CHECK-NEXT: %a      unknown     miss     completed
  %a = arith.addi %u, %c1 : i32
  // ...so an identical sum gets its own entry.
  // CHECK-NEXT: %b      unknown     miss     completed
  %b = arith.addi %u, %c1 : i32
  // The constants are free, so the sum takes the only step. With no steps,
  // the lookup completes and then evaluation runs out.
  // CHECK-NEXT: %s      3           miss     completed
  // NONE:       %s      —           miss     exhausted
  %s = arith.addi %c1, %c2 : i32
  // The key reaches only constants, so the hit is free.
  // CHECK-NEXT: %h      3           hit      completed
  %h = arith.addi %c2, %c1 : i32
  // CHECK-NEXT: %t      7           miss     completed
  %t = arith.addi %c3, %c4 : i32
  // The key reaches %s and %t, which hit for free. With no steps, %s was
  // never cached, so hashing runs out before a lookup.
  // CHECK-NEXT: %f      10          miss     completed
  // NONE:       %f      —           —        exhausted
  %f = arith.addi %s, %t : i32
  return
}
