// Every operation is queried for the first time, so each lookup misses.
// Constants bypass the cache.

func.func @main() {
  %c6 = arith.constant 6 : i32
  %c4 = arith.constant 4 : i32
  %sum = arith.addi %c6, %c4 : i32
  %prod = arith.muli %c6, %c4 : i32
  %diff = arith.subi %c6, %c4 : i32
  return
}
