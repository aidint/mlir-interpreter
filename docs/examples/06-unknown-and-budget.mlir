// Each reported value gets 3 steps. Constants are not cached, so a query pays
// a step for each constant it reaches, even to compute the key of a hit.

func.func @main() {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  // arith.divsi has no evaluation function, so %u is unknown and not cached.
  %u = arith.divsi %c4, %c2 : i32
  // An unknown left operand keys the sum by its operation...
  %a = arith.addi %u, %c1 : i32
  // ...so an identical sum gets its own entry.
  %b = arith.addi %u, %c1 : i32
  // 2 steps for the constants and 1 for the sum.
  %s = arith.addi %c1, %c2 : i32
  // The key costs 2 steps, and a hit needs no more.
  %h = arith.addi %c2, %c1 : i32
  // The key reaches 3 constants, so the lookup completes but no step
  // is left for the sum.
  %e = arith.addi %s, %c3 : i32
  // 2 steps for the constants and 1 for the sum.
  %t = arith.addi %c3, %c4 : i32
  // The key would reach 4 constants, so hashing runs out before a lookup.
  %f = arith.addi %s, %t : i32
  return
}
