// RUN: interpreter %s | FileCheck %s --match-full-lines

// A known zero left operand decides a product without the right one, which a
// cache lookup would evaluate, so such a product isn't cached. Any other
// product is keyed by both operands.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %c0     0           —        completed
  %c0 = arith.constant 0 : i32
  // CHECK-NEXT: %zero   0           —        completed
  %zero = arith.constant 0 : i32
  // CHECK-NEXT: %c7     7           —        completed
  %c7 = arith.constant 7 : i32
  // CHECK-NEXT: %c9     9           —        completed
  %c9 = arith.constant 9 : i32
  // %c7 is never queried.
  // CHECK-NEXT: %p      0           —        completed
  %p = arith.muli %c0, %c7 : i32
  // CHECK-NEXT: %q      0           —        completed
  %q = arith.muli %zero, %c9 : i32
  // A zero right operand is keyed with the left one.
  // CHECK-NEXT: %r      0           miss     completed
  %r = arith.muli %c7, %c0 : i32
  // CHECK-NEXT: %s      63          miss     completed
  %s = arith.muli %c7, %c9 : i32
  return
}
