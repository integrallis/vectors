import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../../main/resources/static/projector/panel-projector.js', import.meta.url), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

function fixture() {
  const requests = [], streams = [], statuses = [], points = [];
  const context = vm.createContext({
    setTimeout, clearTimeout,
    fetch: (url, options) => new Promise(resolve => requests.push({ url, options, resolve })),
    EventSource: class {
      constructor(url) { this.url = url; this.closed = false; streams.push(this); }
      close() { this.closed = true; }
    },
  });
  vm.runInContext(source.replace('export function', 'function'), context);
  const panel = context.createProjectorPanel({
    root: { querySelectorAll: () => [], querySelector: () => null }, collection: 'docs',
    onStatus: s => statuses.push(s), onPoints: p => points.push(p),
  });
  const respond = (request, jobId) => request.resolve({ ok: true, json: async () => ({ jobId, n: 2 }) });
  return { panel, requests, streams, statuses, points, respond };
}

test('a late submit response cannot replace the newest projection', async () => {
  const f = fixture();
  const first = f.panel.run();
  await flush();
  const second = f.panel.run();
  await flush();
  f.respond(f.requests[1], 'new'); await second;
  f.respond(f.requests[0], 'old'); await first;
  assert.equal(f.streams.filter(s => !s.closed).length, 1);
  assert.match(f.streams.find(s => !s.closed).url, /\/new\/events$/);
  assert.ok(f.requests.some(r => r.url.endsWith('/old') && r.options.method === 'DELETE'));
});

test('queued events from a closed stream cannot close or redraw the current projection', async () => {
  const f = fixture();
  const first = f.panel.run(); await flush(); f.respond(f.requests[0], 'old'); await first;
  const old = f.streams[0];
  const second = f.panel.run(); await flush();
  const cancel = f.requests.find(r => r.options.method === 'DELETE');
  if (cancel) cancel.resolve({ ok: true });
  await flush();
  f.respond(f.requests.find(r => r.options.method === 'POST' && r !== f.requests[0]), 'new');
  await second;
  old.onmessage({ data: JSON.stringify({ result: { coords: [[99, 99]], durationMs: 1 } }) });
  old.onerror();
  assert.equal(f.streams[1].closed, false);
  assert.equal(f.points.length, 0);
  assert.match(f.statuses.at(-1), /^running/);
});

test('a completed projection retains its done status when the transport closes', async () => {
  const f = fixture();
  const run = f.panel.run(); await flush(); f.respond(f.requests[0], 'job'); await run;
  const stream = f.streams[0];
  stream.onmessage({ data: JSON.stringify({ result: { coords: [[1, 2]], durationMs: 20 } }) });
  stream.onerror();
  assert.equal(f.statuses.at(-1), 'done · 20 ms');
  assert.equal(f.points.length, 1);
});
