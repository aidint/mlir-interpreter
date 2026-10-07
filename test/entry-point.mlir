// RUN: interpreter %s --split-input-file --verify-diagnostics

// expected-error @below {{no 'func.func @main' entry point}}
module {
  func.func @f() {
    return
  }
}

// -----

// expected-error @below {{entry point has no body}}
func.func private @main()

// -----

// expected-error @below {{entry point must not take arguments}}
func.func @main(%x: i32) {
  return
}

// -----

// expected-error @below {{control flow is not supported}}
func.func @main() {
  cf.br ^bb1
^bb1:
  return
}

// -----

func.func @main() {
  %true = arith.constant true
  // expected-error @below {{control flow is not supported}}
  scf.if %true {
  }
  return
}

// -----

func.func @one() -> i32 {
  %c1 = arith.constant 1 : i32
  return %c1 : i32
}

func.func @main() {
  // expected-error @below {{calls are not supported}}
  %r = func.call @one() : () -> i32
  return
}
