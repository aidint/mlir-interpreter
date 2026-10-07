// RUN: interpreter %s --allow-unregistered-dialect | FileCheck %s --match-full-lines

// Rows use the source names of results, numbering those in a group.

// CHECK: value   evaluated   cache    status
func.func @main() {
  // CHECK-NEXT: %pair#0 unknown     —        completed
  // CHECK-NEXT: %pair#1 unknown     —        completed
  %pair:2 = "test.pair"() : () -> (i32, i32)
  // CHECK-NEXT: %c1     1           —        completed
  %c1 = arith.constant 1 : i32
  // CHECK-NEXT: %sum    unknown     miss     completed
  %sum = arith.addi %pair#1, %c1 : i32
  return
}
