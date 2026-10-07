// Runs the editor's MLIR through the interpreter module in a dedicated worker
// and renders its report. Cancel terminates the worker and starts a fresh one,
// since a running Wasm call can't be interrupted from JavaScript.
'use strict';

const $ = (id) => document.getElementById(id);
const source = $('source');
const budgetInput = $('budget');
const exampleSelect = $('example');
const runButton = $('run');
const cancelButton = $('cancel');
const queryForm = $('query-form');
const queryName = $('query-name');
const queryButton = $('query');
const statusLine = $('status');
const report = $('report');
const rowsBody = $('rows');
const queryBodies = { before: $('query-before'), after: $('query-after') };
const diagnosticsList = $('diagnostics');
const empty = $('empty');
const summary = $('summary');

// Must match the column filler the tool prints.
const NONE = '—';
const MAX_BUDGET = (1n << 64n) - 1n;

let worker = null;
let ready = false;
// The ID of the run in flight, or 0.
let runningId = 0;
let lastId = 0;
// Set when an example asks to run before the module is ready.
let runWhenReady = false;

function setStatus(text, state) {
  statusLine.textContent = text;
  statusLine.dataset.state = state;
}

function updateButtons() {
  runButton.disabled = !ready || runningId !== 0;
  queryButton.disabled = runButton.disabled;
  cancelButton.disabled = runningId === 0;
}

function startWorker() {
  ready = false;
  updateButtons();
  worker = new Worker('worker.js');
  worker.onmessage = ({ data }) => handlers[data.type](data);
  worker.onerror = (event) => {
    event.preventDefault();
    const message = event.message || 'worker error';
    if (!ready)
      handlers['load-error']({ message });
    else if (runningId)
      handlers.crash({ id: runningId, message });
    else
      restartWorker();
  };
}

function restartWorker() {
  worker.terminate();
  startWorker();
}

const handlers = {
  ready({ loadMs }) {
    ready = true;
    updateButtons();
    if (statusLine.dataset.state === 'loading')
      setStatus(`Interpreter ready, loaded in ${loadMs.toFixed(0)} ms.`, 'ready');
    if (runWhenReady)
      run();
  },
  'load-error'({ message }) {
    ready = false;
    runWhenReady = false;
    updateButtons();
    setStatus(`Couldn't load the interpreter: ${message}`, 'error');
  },
  result({ id, report: result, runMs }) {
    if (id !== runningId)
      return;
    finishRun();
    render(result);
    const errors = result.diagnostics.filter((d) => d.severity === 'error');
    if (errors.length)
      setStatus(`Failed with ${plural(errors.length, 'error')}.`, 'error');
    else
      setStatus(`Finished in ${formatMs(runMs)}.`, 'done');
  },
  crash({ id, message }) {
    if (id !== runningId)
      return;
    finishRun();
    render({ rows: [], query: null, diagnostics: [] });
    setStatus(`The interpreter crashed (${message}) and was restarted.`, 'error');
    restartWorker();
  },
};

function finishRun() {
  runningId = 0;
  report.classList.remove('stale');
  updateButtons();
}

function plural(count, noun, nouns = `${noun}s`) {
  return `${count} ${count === 1 ? noun : nouns}`;
}

function formatMs(ms) {
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${ms.toFixed(0)} ms`;
}

function readBudget() {
  const text = budgetInput.value.trim();
  const valid = /^\d+$/.test(text) && BigInt(text) <= MAX_BUDGET;
  budgetInput.setAttribute('aria-invalid', String(!valid));
  return valid ? text : null;
}

// Runs `@main`, and also `query` ({name, before}) if given.
function run(query = null) {
  if (!ready) {
    runWhenReady = true;
    return;
  }
  runWhenReady = false;
  if (runningId)
    return;
  const budget = readBudget();
  if (budget === null) {
    setStatus('The budget must be an integer from 0 to 2⁶⁴ − 1.', 'error');
    budgetInput.focus();
    return;
  }
  runningId = ++lastId;
  updateButtons();
  report.classList.add('stale');
  setStatus(query ? `Running with a query of ${query.name}…` : 'Running…',
            'running');
  worker.postMessage({ id: runningId, source: source.value, budget, query });
}

function runQuery() {
  let name = queryName.value.trim();
  if (!name) {
    setStatus('Enter the name of a result to query, like %b.', 'error');
    queryName.focus();
    return;
  }
  if (!name.startsWith('%'))
    name = `%${name}`;
  queryName.value = name;
  const before = queryForm.elements['query-position'].value === 'before';
  run({ name, before });
}

function cancel() {
  if (!runningId)
    return;
  finishRun();
  setStatus('Cancelled. The interpreter was restarted.', 'cancelled');
  restartWorker();
}

function cell(text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className)
    td.className = className;
  return td;
}

function badge(text, kind) {
  const td = document.createElement('td');
  const span = document.createElement('span');
  span.className = `badge ${kind}`;
  span.textContent = text;
  td.append(span);
  return td;
}

function renderRow(row) {
  const tr = document.createElement('tr');
  let evaluatedClass = '';
  if (row.evaluated === NONE)
    evaluatedClass = 'muted';
  else if (row.evaluated === 'unknown')
    evaluatedClass = 'unknown';
  // The cache column shows the lookup outcome the runner reported, never one
  // inferred from the value.
  const cacheKind = { hit: 'hit', miss: 'miss' }[row.cache] ?? 'none';
  const statusKind = row.status === 'completed' ? 'completed' : 'problem';
  tr.append(cell(row.name, 'name'), cell(row.evaluated, evaluatedClass),
            badge(row.cache, cacheKind), badge(row.status, statusKind));
  return tr;
}

function renderDiagnostic(diag) {
  const li = document.createElement('li');
  li.className = diag.severity;
  if (diag.line) {
    const where = document.createElement('button');
    where.className = 'where';
    where.textContent = `${diag.line}:${diag.column}`;
    where.title = 'Show in the editor';
    where.onclick = () => selectLine(diag.line);
    li.append(where);
  }
  const severity = document.createElement('span');
  severity.className = 'severity';
  severity.textContent = `${diag.severity}:`;
  li.append(severity, diag.message);
  return li;
}

// Renders the direct query in its own group, above or below the rows in the
// order it was made.
function renderQuery(query) {
  for (const body of Object.values(queryBodies))
    body.replaceChildren();
  if (!query)
    return;
  const label = document.createElement('tr');
  const td = cell(`Direct query, ${query.position} the run`, 'label');
  td.colSpan = 4;
  label.append(td);
  const row = renderRow(query);
  row.className = 'query';
  queryBodies[query.position].append(label, row);
}

function render({ rows, query, diagnostics }) {
  rowsBody.replaceChildren(...rows.map(renderRow));
  renderQuery(query);
  diagnosticsList.replaceChildren(...diagnostics.map(renderDiagnostic));
  empty.hidden = rows.length > 0 || diagnostics.length > 0;
  if (!rows.length) {
    summary.textContent = '';
    return;
  }
  const count = (key, value) => rows.filter((row) => row[key] === value).length;
  const parts = [plural(rows.length, 'value'),
                 plural(count('cache', 'hit'), 'hit'),
                 plural(count('cache', 'miss'), 'miss', 'misses')];
  const incomplete = rows.length - count('status', 'completed');
  if (incomplete)
    parts.push(`${incomplete} incomplete`);
  summary.textContent = parts.join(' · ');
}

function selectLine(line) {
  const lines = source.value.split('\n');
  const index = Math.min(line, lines.length) - 1;
  let start = 0;
  for (let i = 0; i < index; ++i)
    start += lines[i].length + 1;
  source.focus();
  source.setSelectionRange(start, start + lines[index].length);
  const lineHeight = parseFloat(getComputedStyle(source).lineHeight);
  source.scrollTop = Math.max(0, (index - 3) * lineHeight);
}

function exampleTitle(name) {
  const words = name.replace(/^\d+-/, '').replace(/-/g, ' ');
  return words[0].toUpperCase() + words.slice(1);
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return response.text();
}

async function loadExample(example) {
  try {
    source.value = await fetchText(`examples/${example.name}.mlir`);
  } catch (error) {
    setStatus(`Couldn't load the example: ${error.message}`, 'error');
    return;
  }
  source.scrollTop = 0;
  budgetInput.value = example.budget;
  run();
}

async function loadExamples() {
  let examples;
  try {
    examples = JSON.parse(await fetchText('examples/index.json'));
  } catch (error) {
    setStatus(`Couldn't load the examples: ${error.message}`, 'error');
    return;
  }
  exampleSelect.replaceChildren(...examples.map((example, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = exampleTitle(example.name);
    return option;
  }));
  exampleSelect.onchange = () => loadExample(examples[exampleSelect.value]);
  if (examples.length)
    await loadExample(examples[0]);
}

runButton.onclick = () => run();
queryForm.onsubmit = (event) => {
  event.preventDefault();
  if (!queryButton.disabled)
    runQuery();
};
cancelButton.onclick = cancel;
budgetInput.oninput = readBudget;
source.onkeydown = (event) => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    if (!runButton.disabled)
      run();
  }
};

startWorker();
loadExamples();
