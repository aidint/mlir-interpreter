// RUN: interpreter %s | FileCheck %s --match-full-lines

// Changing an operand value, the operation kind, the result type or the
// overflow flags changes the key, so each of these lookups misses.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c2     2           —        completed
  %c2 = arith.constant 2 : i32
  // CHECK-NEXT: %c3     3           —        completed
  %c3 = arith.constant 3 : i32
  // CHECK-NEXT: %c4     4           —        completed
  %c4 = arith.constant 4 : i32
  // CHECK-NEXT: %c2_i64 2           —        completed
  %c2_i64 = arith.constant 2 : i64
  // CHECK-NEXT: %c3_i64 3           —        completed
  %c3_i64 = arith.constant 3 : i64
  // First 2 + 3.
  // CHECK-NEXT: %a      5           miss     completed
  %a = arith.addi %c2, %c3 : i32
  // An operand value changed.
  // CHECK-NEXT: %b      6           miss     completed
  %b = arith.addi %c2, %c4 : i32
  // The operation kind changed.
  // CHECK-NEXT: %c      6           miss     completed
  %c = arith.muli %c2, %c3 : i32
  // The result type changed; the value is the same.
  // CHECK-NEXT: %d      5           miss     completed
  %d = arith.addi %c2_i64, %c3_i64 : i64
  // The overflow flags changed.
  // CHECK-NEXT: %e      5           miss     completed
  %e = arith.addi %c2, %c3 overflow<nsw> : i32
  // Same flags as %e, so this hits.
  // CHECK-NEXT: %f      5           hit      completed
  %f = arith.addi %c3, %c2 overflow<nsw> : i32
  return
}
