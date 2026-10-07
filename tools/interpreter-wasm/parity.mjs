// Checks that the Wasm module in a showcase directory reports like the native
// tool, given the same file, budget and named query: every packaged example
// must print the same tables, and malformed inputs must give the same
// diagnostics.
//
//   node parity.mjs docs build/release/tools/interpreter/interpreter
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';

const [docs, native] = process.argv.slice(2).map((arg) => path.resolve(arg));
const createInterpreter =
    createRequire(import.meta.url)(path.join(docs, 'interpreter.js'));
const interpreter = await createInterpreter();

function runWasm(source, budget, query) {
  const sourcePtr = interpreter.stringToNewUTF8(source);
  const namePtr = query ? interpreter.stringToNewUTF8(query.name) : 0;
  const reportPtr = interpreter._interpreter_run(
      sourcePtr, BigInt(budget), namePtr, query?.before ? 1 : 0);
  interpreter._free(sourcePtr);
  interpreter._free(namePtr);
  const report = JSON.parse(interpreter.UTF8ToString(reportPtr));
  interpreter._interpreter_free_report(reportPtr);
  return report;
}

function runNative(file, budget, query) {
  const args = [`--budget=${budget}`, file];
  if (query)
    args.push(`--query=${query.name}`, ...(query.before ? ['--query-before'] : []));
  const result = spawnSync(native, args, { encoding: 'utf8' });
  // Keep only the `file:line:col: severity: message` lines, not the snippets.
  const diagnostics = [...result.stderr.matchAll(
      /^.*:(\d+):(\d+): (error|warning|note|remark): (.*)$/gm)].map(
      ([, line, column, severity, message]) =>
          ({ severity, line: Number(line), column: Number(column), message }));
  return { table: result.stdout, diagnostics };
}

// Mirrors `printColumn` in tools/interpreter/interpreter.cpp.
function column(text, width) {
  return text + ' '.repeat(Math.max(width - [...text].length, 1));
}

function table(header, rows) {
  const line = (name, evaluated, cache, status) =>
      column(name, 8) + column(evaluated, 12) + column(cache, 9) + status + '\n';
  return line(header, 'evaluated', 'cache', 'status') +
         rows.map((row) =>
             line(row.name, row.evaluated, row.cache, row.status)).join('');
}

// Mirrors `run` in tools/interpreter/interpreter.cpp: the query gets its own
// table, printed in the order it was made.
function tables({ rows, query, diagnostics }) {
  if (diagnostics.some((diag) => diag.severity === 'error'))
    return '';
  const main = table('value', rows);
  if (!query)
    return main;
  const queried = table('query', [query]);
  return query.position === 'before' ? `${queried}\n${main}`
                                     : `${main}\n${queried}`;
}

let failures = 0;
function compare(label, file, budget, query) {
  const expected = runNative(file, budget, query);
  const report = runWasm(readFileSync(file, 'utf8'), budget, query);
  const actual = { table: tables(report), diagnostics: report.diagnostics };
  if (JSON.stringify(actual) === JSON.stringify(expected)) {
    console.log(`ok   ${label} (budget ${budget}, ${report.rows.length} rows, ` +
                `${report.diagnostics.length} diagnostics)`);
    return;
  }
  ++failures;
  console.log(`FAIL ${label}\n--- native\n${JSON.stringify(expected, null, 2)}` +
              `\n--- wasm\n${JSON.stringify(actual, null, 2)}`);
}

const examples =
    JSON.parse(readFileSync(path.join(docs, 'examples/index.json'), 'utf8'));
for (const { name, budget } of examples)
  compare(name, path.join(docs, 'examples', `${name}.mlir`), budget);

// The same name before and after the rows, at a budget that exhausts some of
// them and at one that doesn't, and a name that doesn't exist.
const equalValues = path.join(docs, 'examples', '02-equal-values.mlir');
for (const budget of [3, 1000]) {
  for (const before of [true, false]) {
    compare(`02-equal-values --query=%b${before ? ' --query-before' : ''}`,
            equalValues, budget, { name: '%b', before });
  }
}
compare('02-equal-values --query=%nope', equalValues, 1000,
        { name: '%nope', before: false });

const malformed = {
  'undeclared-value': 'func.func @main() {\n  %a = arith.addi %b, %b : i32\n  return\n}\n',
  'type-mismatch': 'func.func @main() {\n  %c = arith.constant 1 : i32\n  %a = arith.addi %c, %c : i64\n  return\n}\n',
  'no-main': 'func.func @f() {\n  return\n}\n',
  'external-main': 'func.func private @main()\n',
};
const scratch = mkdtempSync(path.join(os.tmpdir(), 'parity-'));
try {
  for (const [name, source] of Object.entries(malformed)) {
    const file = path.join(scratch, `${name}.mlir`);
    writeFileSync(file, source);
    compare(name, file, 1000);
  }
  // Block arguments, of @main and of a nested block, can be queried by name.
  const blockArguments = path.join(scratch, 'block-arguments.mlir');
  writeFileSync(blockArguments, 'func.func @main(%arg1: i32) {\n  %c1 = arith.constant 1 : i32\n  %a = arith.addi %arg1, %c1 : i32\n  cf.br ^bb1(%a : i32)\n^bb1(%x: i32):\n  return\n}\n');
  for (const name of ['%arg1', '%x']) {
    for (const before of [true, false]) {
      compare(`block-arguments --query=${name}${before ? ' --query-before' : ''}`,
              blockArguments, 1000, { name, before });
    }
  }
} finally {
  rmSync(scratch, { recursive: true });
}

process.exit(failures ? 1 : 0);
