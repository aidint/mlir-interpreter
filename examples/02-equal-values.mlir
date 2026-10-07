// RUN: interpreter %s | FileCheck %s --match-full-lines

// Keys hold evaluated operand values, not SSA operands, so operations with
// distinct operands of equal values share an entry.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c2     2           —        completed
  %c2 = arith.constant 2 : i32
  // CHECK-NEXT: %c3     3           —        completed
  %c3 = arith.constant 3 : i32
  // CHECK-NEXT: %two    2           —        completed
  %two = arith.constant 2 : i32
  // CHECK-NEXT: %three  3           —        completed
  %three = arith.constant 3 : i32
  // CHECK-NEXT: %c5     5           —        completed
  %c5 = arith.constant 5 : i32
  // First 2 + 3.
  // CHECK-NEXT: %a      5           miss     completed
  %a = arith.addi %c2, %c3 : i32
  // Other constants with the same values.
  // CHECK-NEXT: %b      5           hit      completed
  %b = arith.addi %two, %three : i32
  // First 5 * 2, with 5 computed by %a.
  // CHECK-NEXT: %c      10          miss     completed
  %c = arith.muli %a, %c2 : i32
  // The constant 5 equals the computed %a.
  // CHECK-NEXT: %d      10          hit      completed
  %d = arith.muli %c5, %two : i32
  // First 5 - 3.
  // CHECK-NEXT: %e      2           miss     completed
  %e = arith.subi %a, %c3 : i32
  // 5 - 3 again.
  // CHECK-NEXT: %f      2           hit      completed
  %f = arith.subi %c5, %three : i32
  return
}
