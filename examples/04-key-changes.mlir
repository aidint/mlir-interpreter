// Changing an operand value, the operation kind, the result type or the
// overflow flags changes the key, so each of these lookups misses.
func.func @main() {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c2_i64 = arith.constant 2 : i64
  %c3_i64 = arith.constant 3 : i64
  // miss: 2 + 3.
  %a = arith.addi %c2, %c3 : i32
  // miss: an operand value changed.
  %b = arith.addi %c2, %c4 : i32
  // miss: the operation kind changed.
  %c = arith.muli %c2, %c3 : i32
  // miss: the result type changed; the value is the same.
  %d = arith.addi %c2_i64, %c3_i64 : i64
  // miss: the overflow flags changed.
  %e = arith.addi %c2, %c3 overflow<nsw> : i32
  // hit: same flags as %e.
  %f = arith.addi %c3, %c2 overflow<nsw> : i32
  return
}
