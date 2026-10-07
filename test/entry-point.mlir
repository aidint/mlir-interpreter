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
