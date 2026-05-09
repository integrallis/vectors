// Vectors Studio — live progress for an optimization study. Subscribes to the SSE event stream,
// appends a row per completed trial, and redraws a score-vs-trial line chart with running-best.
import { drawHistoryChart } from "./chart.js";
import { renderTrialRow, sortTrialTable } from "./table.js";

const shell = document.getElementById("optimize-progress");
if (!shell) throw new Error("optimize/progress: shell not found");

const studyId = shell.dataset.studyId;
const collection = shell.dataset.collection;
const startedAt = Date.now();

const state = {
  trials: [],
  trialsTotal: 0,
  state: "PENDING",
};

const bind = (name) => shell.querySelector(`[data-bind=${name}]`);
const setText = (name, text) => {
  const el = bind(name);
  if (el) el.textContent = text;
};

const tickElapsed = () => {
  const elapsed = Math.round((Date.now() - startedAt) / 1000);
  const m = Math.floor(elapsed / 60);
  const s = elapsed % 60;
  setText("elapsed", m > 0 ? `${m}m ${s}s` : `${s}s`);
};
setInterval(tickElapsed, 1000);
tickElapsed();

const refresh = () => {
  setText("trials-completed", String(state.trials.length));
  setText("trials-total", String(state.trialsTotal));
  setText("state", state.state);
  if (state.trials.length > 0) {
    const best = state.trials.reduce(
      (a, b) => (a.objectiveScore >= b.objectiveScore ? a : b)
    );
    setText("best-score", best.objectiveScore.toFixed(4));
  }
  drawHistoryChart(document.getElementById("history-canvas"), state.trials);
};

const onTrial = (ev) => {
  const idx = ev.index ?? state.trials.length;
  while (state.trials.length <= idx) state.trials.push(null);
  state.trials[idx] = ev.result;
  const tbody = document.getElementById("trial-rows");
  tbody.appendChild(renderTrialRow(idx, ev.result, collection, studyId));
  refresh();
};

const sse = new EventSource(`/api/optimize/studies/${studyId}/events`);
sse.onmessage = (msg) => {
  let ev;
  try { ev = JSON.parse(msg.data); } catch { return; }
  if (Object.prototype.hasOwnProperty.call(ev, "result") && ev.result) onTrial(ev);
  else if (typeof ev.trialsTotal === "number") {
    state.trialsTotal = ev.trialsTotal;
    if (typeof ev.trialsCompleted === "number") {
      // ProgressEvt: ensure stats stay in sync even when trials replay arrives early.
    }
    refresh();
  } else if (typeof ev.nTrials === "number") {
    state.trialsTotal = ev.nTrials;
    state.state = "RUNNING";
    refresh();
  } else if (Object.prototype.hasOwnProperty.call(ev, "trialsCompleted") && Object.keys(ev).length === 2) {
    // CompletedEvt or CancelledEvt — we can't distinguish without a tag; treat as terminal.
    state.state = "COMPLETED";
    sse.close();
    refresh();
  } else if (Object.prototype.hasOwnProperty.call(ev, "message")) {
    state.state = "FAILED";
    sse.close();
    refresh();
  }
};
sse.onerror = () => sse.close();

// Cancel
shell.querySelector("[data-action=cancel]")?.addEventListener("click", async () => {
  await fetch(`/api/optimize/studies/${studyId}/cancel`, { method: "POST" });
});

// Sortable table
shell.querySelectorAll("th[data-sort]").forEach((th) => {
  th.addEventListener("click", () => sortTrialTable(th));
});
