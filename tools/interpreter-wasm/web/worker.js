// Loads the interpreter module once, then runs each request in it. Every run
// gets a fresh MLIR context and cache inside the module; the module itself is
// kept for the next run.
'use strict';

let interpreter = null;

const loaded = (async () => {
  const start = performance.now();
  // `interpreter.wasm` is resolved next to this script, so relative URLs keep
  // working under any path prefix.
  importScripts('interpreter.js');
  interpreter = await createInterpreter();
  postMessage({ type: 'ready', loadMs: performance.now() - start });
})().catch((error) => {
  postMessage({ type: 'load-error', message: String(error) });
});

function run(source, budget) {
  const sourcePtr = interpreter.stringToNewUTF8(source);
  let reportPtr;
  try {
    reportPtr = interpreter._interpreter_run(sourcePtr, BigInt(budget));
  } finally {
    interpreter._free(sourcePtr);
  }
  try {
    return JSON.parse(interpreter.UTF8ToString(reportPtr));
  } finally {
    interpreter._interpreter_free_report(reportPtr);
  }
}

onmessage = async ({ data }) => {
  await loaded;
  if (!interpreter)
    return;
  const { id, source, budget } = data;
  const start = performance.now();
  try {
    const report = run(source, budget);
    postMessage({ type: 'result', id, report, runMs: performance.now() - start });
  } catch (error) {
    // A trap or abort leaves the module unusable, so the page replaces this
    // worker.
    postMessage({ type: 'crash', id, message: String(error) });
  }
};
