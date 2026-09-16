func.func @f(%arg0: index) -> index {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %a = arith.muli %c3, %c4 : index
  %b = arith.addi %arg0, %c1 : index
  %c = arith.addi %a, %b : index
  %d = arith.muli %a, %c2 : index
  return %d : index
}
