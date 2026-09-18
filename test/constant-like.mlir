func.func @f() -> index {
  %a = index.constant 6
  %b = index.constant 7
  %c = arith.muli %a, %b : index
  return %c : index
}

func.func @g() -> i1 {
  %t = index.bool.constant true
  return %t : i1
}
