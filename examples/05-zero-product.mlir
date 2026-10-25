// RUN: interpreter %s | FileCheck %s --match-full-lines

// A known zero decides a product, so its key holds only the zero and never
// queries the other operand. Every product with a zero shares one entry.

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
  // CHECK-NEXT: %p      0           miss     completed
  %p = arith.muli %c0, %c7 : i32
  // Another zero and another operand share the zero-product entry.
  // CHECK-NEXT: %q      0           hit      completed
  %q = arith.muli %zero, %c9 : i32
  // A zero right operand also gives the zero-product key.
  // CHECK-NEXT: %r      0           hit      completed
  %r = arith.muli %c7, %c0 : i32
  // A nonzero product needs both operands.
  // CHECK-NEXT: %s      63          miss     completed
  %s = arith.muli %c7, %c9 : i32
  return
}
