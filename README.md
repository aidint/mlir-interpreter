# interpreter

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

```sh
cmake --build --preset debug --target check-interpreter
```

Tests run through `lit` and `FileCheck` from the submodule, like MLIR's own.
To run one test:

```sh
build/debug/third_party/llvm-project/llvm/bin/llvm-lit build/debug/examples/05-zero-product.mlir
```

## Layout

- `include/interpreter/`, `lib/`:
  - `InterpreterInterfaces`: `EvaluableOpInterface`, `Answer`, `EvalContext`.
  - `Interpreter`: the engine.
- `include/interpreter/Dialects/`, `lib/Dialects/`, one library per dialect:
  - `InterpreterBuiltin`: builtin values and attribute models
    (`registerBuiltinEvalValues`).
  - `InterpreterArith`: `arith` external models (`registerArithEvalExternalModels`).
- `include/interpreter/Tools/`, `lib/Tools/`: `InterpreterRunner`, which runs
  `@main` and builds the report (`runMain`) for the tool and the Wasm module.
- `tools/interpreter/`: the `interpreter` tool.
- `tools/interpreter-wasm/`: the runner as a WebAssembly module, and the
  showcase page.
- `examples/`: MLIR modules for the tool. Each checks its own report with
  `RUN` and `CHECK` lines.
- `test/`: the shared `lit` config, tool tests, and the C++ regressions
  `engine-lifetime` (allocator lifetimes) and `eval-cache` (the cache).

## interpreter

Runs `func.func @main` of an MLIR file. It queries every result in `@main`,
across all blocks and nested regions in source order, through one engine, so
all queries share one cache. It prints one row per result:

- `evaluated`: the value, `unknown`, or `—` when the query didn't complete.
- `cache`: the outcome of the lookup for that result's own operation, not for
  nested queries. It shows `hit` or `miss` when the lookup completes, and `—`
  when the operation isn't cached (constants, operations without an evaluation
  function) or hashing ran out first. A query can miss and then exhaust.
- `status`: `completed`, `exhausted`, `needs order` or `cycle`.

Values are queried on demand, so a row doesn't depend on which branch would
run. Block arguments, including `@main`'s, and results of ops without an
evaluation function (`scf.if`, `func.call`, ...) are `unknown`. The tool reports
an error only when there is no `@main` or it has no body.

`--budget=N` (default 1000) is the step budget of each reported value: it is
reset for every row, while the cache is kept for the whole run. Hashing,
equality and evaluation share that budget. A hit can still spend budget
computing its key, since keys hold evaluated operands, and constants are not
cached across queries, so each query pays a step per constant it reaches.

`--query=NAME` also queries the value with that source name, a result like
`%b` or a block argument like `%arg0`, through the same engine, so it shares the
rows' cache. It runs after the rows,
or before them with `--query-before`, and prints its own table in that order:

```sh
$ build/debug/tools/interpreter/interpreter --query=%b --query-before test/named-query.mlir
query   evaluated   cache    status
%b      5           miss     completed

value   evaluated   cache    status
...
%a      5           hit      completed
%b      5           hit      completed
```

Other options: `--allow-unregistered-dialect`, and `--split-input-file` and
`--verify-diagnostics`, which work as in `mlir-opt`.

### Examples

`arith.addi`, `arith.muli` and `arith.subi` are keyed by their kind, result
type, overflow flags and evaluated operands. Addition and multiplication also
match swapped operands. Each example explains its rows in comments.

```sh
$ build/debug/tools/interpreter/interpreter examples/01-basic.mlir
value   evaluated   cache    status
%c6     6           —        completed
%c4     4           —        completed
%sum    10          miss     completed
%prod   24          miss     completed
%diff   2           miss     completed
```

Equal operand values hit, even when one is computed and the other a constant:

```sh
$ build/debug/tools/interpreter/interpreter examples/02-equal-values.mlir
value   evaluated   cache    status
%c2     2           —        completed
%c3     3           —        completed
%two    2           —        completed
%three  3           —        completed
%c5     5           —        completed
%a      5           miss     completed
%b      5           hit      completed
%c      10          miss     completed
%d      10          hit      completed
%e      2           miss     completed
%f      2           hit      completed
```

Swapped operands hit for addition and multiplication, but not subtraction:

```sh
$ build/debug/tools/interpreter/interpreter examples/03-swapped-operands.mlir
value   evaluated   cache    status
%c2     2           —        completed
%c3     3           —        completed
%two    2           —        completed
%a      5           miss     completed
%b      5           hit      completed
%c      6           miss     completed
%d      6           hit      completed
%e      1           miss     completed
%f      -1          miss     completed
%g      0           miss     completed
%h      0           hit      completed
```

Changing an operand value, the kind, the result type or the overflow flags
misses:

```sh
$ build/debug/tools/interpreter/interpreter examples/04-key-changes.mlir
value   evaluated   cache    status
%c2     2           —        completed
%c3     3           —        completed
%c4     4           —        completed
%c2_i64 2           —        completed
%c3_i64 3           —        completed
%a      5           miss     completed
%b      6           miss     completed
%c      6           miss     completed
%d      5           miss     completed
%e      5           miss     completed
%f      5           hit      completed
```

A known zero keys a product without its other operand, so `%p` fits in 2 steps
and every product with a zero shares one entry:

```sh
$ build/debug/tools/interpreter/interpreter --budget=2 examples/05-zero-product.mlir
value   evaluated   cache    status
%c0     0           —        completed
%zero   0           —        completed
%c7     7           —        completed
%c9     9           —        completed
%p      0           miss     completed
%q      0           hit      completed
%r      0           hit      completed
%s      —           miss     exhausted
```

Unknown operands fall back to the operation's identity, and a small budget
runs out either after the lookup (`miss`) or during hashing (`—`):

```sh
$ build/debug/tools/interpreter/interpreter --budget=3 examples/06-unknown-and-budget.mlir
value   evaluated   cache    status
%c1     1           —        completed
%c2     2           —        completed
%c3     3           —        completed
%c4     4           —        completed
%u      unknown     —        completed
%a      unknown     miss     completed
%b      unknown     miss     completed
%s      3           miss     completed
%h      3           hit      completed
%e      —           miss     exhausted
%t      7           miss     completed
%f      —           —        exhausted
```

The tool registers the `func`, `arith`, `cf`, `scf`, `index` and `ub` dialects.
To support more, add them to `registerRunnerDialects` in `lib/Tools/Runner.cpp`
and link the matching `MLIR*Dialect` libraries in `lib/Tools/CMakeLists.txt`.

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
