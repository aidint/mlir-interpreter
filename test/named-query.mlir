// RUN: interpreter %s --query=%%b | FileCheck %s --check-prefix=AFTER --match-full-lines
// RUN: interpreter %s --query=%%b --query-before | FileCheck %s --check-prefix=BEFORE --match-full-lines
// RUN: interpreter %s --query=%%arg | FileCheck %s --check-prefix=ARG --match-full-lines
// RUN: interpreter %s --query=%%nope --verify-diagnostics

// A named query goes through the same engine as the rows, so it shares their
// cache. After the rows, %b hits %a's entry; before them, %b is the first
// entry, and %a hits it instead. A block argument can be queried too, and is
// unknown.

// AFTER:       value   evaluated   cache    status
// AFTER:       %a      5           miss     completed
// AFTER-NEXT:  %b      5           hit      completed
// AFTER-EMPTY:
// AFTER-NEXT:  query   evaluated   cache    status
// AFTER-NEXT:  %b      5           hit      completed

// BEFORE:      query   evaluated   cache    status
// BEFORE-NEXT: %b      5           miss     completed
// BEFORE-EMPTY:
// BEFORE-NEXT: value   evaluated   cache    status
// BEFORE:      %a      5           hit      completed
// BEFORE-NEXT: %b      5           hit      completed

// ARG:      query   evaluated   cache    status
// ARG-NEXT: %arg    unknown     —        completed

// expected-error @below {{no value named '%nope' in @main}}
func.func @main(%arg: i32) {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %two = arith.constant 2 : i32
  %three = arith.constant 3 : i32
  %a = arith.addi %c2, %c3 : i32
  %b = arith.addi %two, %three : i32
  return
}
