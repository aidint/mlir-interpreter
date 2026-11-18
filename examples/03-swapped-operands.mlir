// RUN: interpreter %s | FileCheck %s --match-full-lines

// Operands are compared in order, so swapped operands miss, even for addition
// and multiplication. Swapping equal operands still hits.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c2     2           —        completed
  %c2 = arith.constant 2 : i32
  // CHECK-NEXT: %c3     3           —        completed
  %c3 = arith.constant 3 : i32
  // CHECK-NEXT: %two    2           —        completed
  %two = arith.constant 2 : i32
  // First 2 + 3.
  // CHECK-NEXT: %a      5           miss     completed
  %a = arith.addi %c2, %c3 : i32
  // 3 + 2 doesn't match 2 + 3.
  // CHECK-NEXT: %b      5           miss     completed
  %b = arith.addi %c3, %c2 : i32
  // First 2 * 3.
  // CHECK-NEXT: %c      6           miss     completed
  %c = arith.muli %c2, %c3 : i32
  // 3 * 2 doesn't match 2 * 3.
  // CHECK-NEXT: %d      6           miss     completed
  %d = arith.muli %c3, %c2 : i32
  // First 3 - 2.
  // CHECK-NEXT: %e      1           miss     completed
  %e = arith.subi %c3, %c2 : i32
  // 2 - 3 is not 3 - 2.
  // CHECK-NEXT: %f      -1          miss     completed
  %f = arith.subi %c2, %c3 : i32
  // First 2 - 2.
  // CHECK-NEXT: %g      0           miss     completed
  %g = arith.subi %c2, %two : i32
  // Swapping equal operands leaves 2 - 2.
  // CHECK-NEXT: %h      0           hit      completed
  %h = arith.subi %two, %c2 : i32
  return
}
