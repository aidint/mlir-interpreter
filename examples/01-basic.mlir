// RUN: interpreter %s | FileCheck %s --match-full-lines

// Every operation is queried for the first time, so each lookup misses.
// Constants bypass the cache.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c6     6           —        completed
  %c6 = arith.constant 6 : i32
  // CHECK-NEXT: %c4     4           —        completed
  %c4 = arith.constant 4 : i32
  // CHECK-NEXT: %sum    10          miss     completed
  %sum = arith.addi %c6, %c4 : i32
  // CHECK-NEXT: %prod   24          miss     completed
  %prod = arith.muli %c6, %c4 : i32
  // CHECK-NEXT: %diff   2           miss     completed
  %diff = arith.subi %c6, %c4 : i32
  return
}
