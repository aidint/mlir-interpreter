// Run: build/debug/tools/mlir-canon/mlir-canon test/canonicalize.mlir
//
// Expected: the additions fold to a single constant 6, and the unused
// multiplication is removed.

func.func @fold() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.addi %c1, %c2 : i32
  %1 = arith.addi %0, %c3 : i32
  %unused = arith.muli %1, %1 : i32
  return %1 : i32
}
