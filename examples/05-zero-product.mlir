// Run with --budget=2.
//
// A known zero decides a product, so its key holds only the zero and never
// queries the other operand. Each reported value gets 2 steps, and constants
// cost a step per query.
func.func @main() {
  %c0 = arith.constant 0 : i32
  %zero = arith.constant 0 : i32
  %c7 = arith.constant 7 : i32
  %c9 = arith.constant 9 : i32
  // miss: 1 step for %c0 and 1 for the product; %c7 is never queried.
  %p = arith.muli %c0, %c7 : i32
  // hit: another zero and another operand share the zero-product entry.
  %q = arith.muli %zero, %c9 : i32
  // hit: a zero right operand also gives the zero-product key.
  %r = arith.muli %c7, %c0 : i32
  // miss: a nonzero product needs both constants, leaving no step for itself.
  %s = arith.muli %c7, %c9 : i32
  return
}
