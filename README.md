# mlir-interpreter

C++/CMake project built with clang++ against MLIR from the
`third_party/llvm-project` submodule (no system LLVM/MLIR is used).

## Setup

```sh
git submodule update --init --depth 1 third_party/llvm-project
cmake --preset debug          # or: release
cmake --build --preset debug
```

Only the LLVM/MLIR libraries our targets link against are built. The first
build takes a few minutes. After that, ccache makes rebuilds fast.

## Layout

- `include/mlir-interpreter/`, `lib/`:
  - `MLIRInterpreterInterfaces`: `EvaluableOpInterface`, `Answer`, `EvalContext`.
  - `MLIRInterpreter`: the engine.
  - `MLIRInterpreterArith`: `arith` external models (`registerArithEvalExternalModels`).
- `tools/mlir-interpreter/`: the `mlir-interpreter` tool.
- `test/`: example MLIR inputs.

## mlir-interpreter

Parses an MLIR file and queries every op result:

```sh
$ build/debug/tools/mlir-interpreter/mlir-interpreter test/query.mlir
%c1 -> 1
...
```

Options: `--budget=<steps>`, `--allow-unregistered-dialect`.

It registers the `func`, `arith`, `cf`, `scf`, `math` and `ub` dialects. To
support more, add them in `tools/mlir-interpreter/mlir-interpreter.cpp` and link
the matching `MLIR*Dialect` libraries in its `CMakeLists.txt`.

## Updating LLVM

The submodule is shallow and tracks `main`:

```sh
git submodule update --remote --depth 1 third_party/llvm-project
git add third_party/llvm-project
```

## clangd

Configuring writes `compile_commands.json` at the repo root, as a symlink to the
build directory you configured last. Build at least once so the generated
`.inc` headers exist. The compile database also covers the LLVM/MLIR sources, so
go-to-definition works inside the submodule.

LLVM's precompiled headers are turned off (`CMAKE_DISABLE_PRECOMPILE_HEADERS`)
because clangd can't read PCH files.
