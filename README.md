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
  - `InterpreterSCF`: the `scf.for` model (`registerSCFEvalExternalModels`).
- `tools/interpreter/`: the `interpreter` tool.
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
  when the operation isn't cached (constants whose attribute doesn't opt in,
  such as integers, and operations without an evaluation function) or hashing
  ran out first. A query can miss and then exhaust.
- `status`: `completed`, `exhausted`, `needs order` or `cycle`.

Values are queried on demand, so a row doesn't depend on which branch would
run. Block arguments are known only while a loop iteration binds them, so
`@main`'s arguments, and rows inside a loop body, are `unknown`, as are results
of ops without an evaluation function (`scf.if`, `func.call`, ...). The tool
reports an error only when there is no `@main` or it has no body.

`--budget=N` (default 1000) is the step budget of each reported value: it is
reset for every row, while the cache is kept for the whole run. Hashing,
equality and evaluation share that budget. Each evaluated operation, and each
loop iteration, costs a step, while constants cost none. A hit can still spend budget computing its
key, since keys hold evaluated operands, but only for operands that are neither
constants nor cached.

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

A known zero keys a product without its other operand, so every product with a
zero shares one entry:

```sh
$ build/debug/tools/interpreter/interpreter examples/05-zero-product.mlir
value   evaluated   cache    status
%c0     0           —        completed
%zero   0           —        completed
%c7     7           —        completed
%c9     9           —        completed
%p      0           miss     completed
%q      0           hit      completed
%r      0           hit      completed
%s      63          miss     completed
```

Unknown operands fall back to the operation's identity. Constants and hits are
free, so 1 step per value is enough when values are queried in order:

```sh
$ build/debug/tools/interpreter/interpreter --budget=1 examples/06-unknown-and-budget.mlir
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
%t      7           miss     completed
%f      10          miss     completed
```

With `--budget=0` only constants and `unknown` values complete. A query runs out
either after the lookup (`miss`, as for `%s`) or during hashing (`—`, as for
`%f`, whose operand `%s` was never cached).

`scf.for` binds its induction variable and iteration arguments for each
iteration, and nested loops iterate inside the outer iteration. Ops in a running
loop skip the cache, since their keys depend on the iteration:

```sh
$ build/debug/tools/interpreter/interpreter examples/07-loops.mlir
value   evaluated   cache    status
%c0     0           —        completed
%c1     1           —        completed
%c3     3           —        completed
%c4     4           —        completed
%sum    6           miss     completed
%next   unknown     miss     completed
%prod   18          miss     completed
%inner  unknown     miss     completed
%ij     unknown     miss     completed
%n      unknown     miss     completed
```

A loop outside any running loop is cached like any other op. It matches a loop
over other SSA values with equal values and the same body structure:

```sh
$ build/debug/tools/interpreter/interpreter examples/08-equivalent-loops.mlir
value   evaluated   cache    status
%c0     0           —        completed
%c1     1           —        completed
%c2     2           —        completed
%c4     4           —        completed
%a      6           miss     completed
%next   unknown     miss     completed
%zero   0           miss     completed
%four   4           miss     completed
%b      6           hit      completed
%t      unknown     miss     completed
%c      0           miss     completed
%u      unknown     miss     completed
```

The tool registers the `func`, `arith`, `cf`, `scf`, `index` and `ub` dialects.
To support more, add them in `tools/interpreter/interpreter.cpp` and link the
matching `MLIR*Dialect` libraries in its `CMakeLists.txt`.

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
