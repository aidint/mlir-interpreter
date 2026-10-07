// Keys hold evaluated operand values, not SSA operands, so operations with
// distinct operands of equal values share an entry.

func.func @main() {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %two = arith.constant 2 : i32
  %three = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  // First 2 + 3.
  %a = arith.addi %c2, %c3 : i32
  // Other constants with the same values.
  %b = arith.addi %two, %three : i32
  // First 5 * 2, with 5 computed by %a.
  %c = arith.muli %a, %c2 : i32
  // The constant 5 equals the computed %a.
  %d = arith.muli %c5, %two : i32
  // First 5 - 3.
  %e = arith.subi %a, %c3 : i32
  // 5 - 3 again.
  %f = arith.subi %c5, %three : i32
  return
}
