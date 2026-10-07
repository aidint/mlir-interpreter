// RUN: interpreter %s | FileCheck %s --match-full-lines

// Addition and multiplication match swapped operands; subtraction keeps its
// operand order.

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
  // 3 + 2 matches 2 + 3.
  // CHECK-NEXT: %b      5           hit      completed
  %b = arith.addi %c3, %c2 : i32
  // First 2 * 3.
  // CHECK-NEXT: %c      6           miss     completed
  %c = arith.muli %c2, %c3 : i32
  // 3 * 2 matches 2 * 3.
  // CHECK-NEXT: %d      6           hit      completed
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
