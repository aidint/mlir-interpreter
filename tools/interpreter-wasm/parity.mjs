// Checks that the Wasm module in a showcase directory reports like the native
// tool, given the same file and budget: every packaged example must print the
// same table, and malformed inputs must give the same diagnostics.
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

function runWasm(source, budget) {
  const sourcePtr = interpreter.stringToNewUTF8(source);
  const reportPtr = interpreter._interpreter_run(sourcePtr, BigInt(budget));
  interpreter._free(sourcePtr);
  const report = JSON.parse(interpreter.UTF8ToString(reportPtr));
  interpreter._interpreter_free_report(reportPtr);
  return report;
}

function runNative(file, budget) {
  const result = spawnSync(native, [`--budget=${budget}`, file],
                           { encoding: 'utf8' });
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

function table(rows) {
  if (!rows.length)
    return '';
  const line = (name, evaluated, cache, status) =>
      column(name, 8) + column(evaluated, 12) + column(cache, 9) + status + '\n';
  return line('value', 'evaluated', 'cache', 'status') +
         rows.map((row) =>
             line(row.name, row.evaluated, row.cache, row.status)).join('');
}

let failures = 0;
function compare(label, file, budget) {
  const expected = runNative(file, budget);
  const report = runWasm(readFileSync(file, 'utf8'), budget);
  const actual = { table: table(report.rows), diagnostics: report.diagnostics };
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
} finally {
  rmSync(scratch, { recursive: true });
}

process.exit(failures ? 1 : 0);
