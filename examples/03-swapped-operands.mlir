// Addition and multiplication match swapped operands; subtraction keeps its
// operand order.
func.func @main() {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %two = arith.constant 2 : i32
  // miss: 2 + 3.
  %a = arith.addi %c2, %c3 : i32
  // hit: 3 + 2.
  %b = arith.addi %c3, %c2 : i32
  // miss: 2 * 3.
  %c = arith.muli %c2, %c3 : i32
  // hit: 3 * 2.
  %d = arith.muli %c3, %c2 : i32
  // miss: 3 - 2.
  %e = arith.subi %c3, %c2 : i32
  // miss: 2 - 3 is not 3 - 2.
  %f = arith.subi %c2, %c3 : i32
  // miss: 2 - 2.
  %g = arith.subi %c2, %two : i32
  // hit: swapping equal operands leaves 2 - 2.
  %h = arith.subi %two, %c2 : i32
  return
}
