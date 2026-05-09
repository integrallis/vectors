// Right-rail Projector panel. Hosts the PCA / t-SNE / UMAP tabs, drives the
// run / pause / re-run lifecycle, and forwards SSE coordinates to the scene.
// Server-side parameter keys live in ApiRoutes.paramsFrom.

const ALGOS = { pca: "PCA", tsne: "TSNE", umap: "UMAP" };

export function createProjectorPanel({ root, collection, onPoints, onStatus }) {
  const tabs = root.querySelectorAll(".projector-tab");
  const tabPanels = root.querySelectorAll(".projector-tab-panel");
  const dimRadios = root.querySelectorAll('input[name="proj-dim"]');
  const runBtn = root.querySelector("#proj-run");
  const pauseBtn = root.querySelector("#proj-pause");
  const rerunBtn = root.querySelector("#proj-rerun");
  const iterEl = root.querySelector("#proj-iter");

  const num = (sel, dflt) => {
    const el = root.querySelector(sel);
    const v = parseFloat(el?.value);
    return Number.isFinite(v) ? v : dflt;
  };
  const intNum = (sel, dflt) => {
    const v = num(sel, dflt);
    return Math.trunc(v);
  };

  let activeTab = "pca";
  let state = "idle"; // idle | running | paused
  let activeJob = null;
  let activeStream = null;
  let lastSubmit = null;

  function selectTab(tab) {
    activeTab = tab;
    tabs.forEach((t) => t.classList.toggle("is-active", t.dataset.tab === tab));
    tabPanels.forEach((p) => p.classList.toggle("is-active", p.dataset.tab === tab));
  }

  function selectedDim() {
    for (const r of dimRadios) if (r.checked) return parseInt(r.value, 10);
    return 3;
  }

  function paramsFor(tab) {
    if (tab === "pca") return { center: true, whiten: false };
    if (tab === "tsne") return {
      perplexity: intNum("#tsne-perplexity", 15),
      learningRate: num("#tsne-learning-rate", 200),
      iterations: intNum("#tsne-iterations", 1000),
      seed: intNum("#tsne-seed", 42),
    };
    if (tab === "umap") return {
      neighbors: intNum("#umap-neighbors", 15),
      minDist: num("#umap-min-dist", 0.1),
      iterations: intNum("#umap-iterations", 200),
      seed: intNum("#umap-seed", 42),
    };
    return {};
  }

  function setIter(i, total) { iterEl.textContent = `iter ${i ?? 0} / ${total ?? 0}`; }

  function applyState(s) {
    state = s;
    runBtn.disabled = s !== "idle";
    pauseBtn.disabled = s === "idle";
    pauseBtn.textContent = s === "paused" ? "Resume" : "Pause";
    rerunBtn.disabled = lastSubmit == null;
  }

  function closeStream() {
    if (activeStream) { activeStream.close(); activeStream = null; }
  }

  function buildBody() {
    return {
      collection,
      algorithm: ALGOS[activeTab],
      dimensions: selectedDim(),
      sampleSize: 0,
      params: paramsFor(activeTab),
      sphereize: false,
    };
  }

  async function submit(body) {
    closeStream();
    lastSubmit = body;
    applyState("running");
    onStatus(`submitting ${body.algorithm}…`);
    setIter(0, 0);
    let resp;
    try {
      resp = await fetch("/api/projections", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) { onStatus(`network error: ${e.message}`); applyState("idle"); return; }
    if (!resp.ok) { onStatus(`submit failed: ${resp.status}`); applyState("idle"); return; }
    const { jobId, n } = await resp.json();
    activeJob = jobId;
    onStatus(`running ${body.algorithm} on ${n} points…`);
    activeStream = new EventSource(`/api/projections/${jobId}/events`);
    activeStream.onmessage = (e) => {
      let ev; try { ev = JSON.parse(e.data); } catch { return; }
      const dim = body.dimensions;
      if (ev.coords) {
        onPoints(ev.coords, dim);
        setIter(ev.iter, ev.total);
        onStatus(`iter ${ev.iter} / ${ev.total}`);
      } else if (ev.result) {
        onPoints(ev.result.coords, dim);
        onStatus(`done · ${ev.result.durationMs} ms`);
        finish();
      } else if (ev.message) {
        onStatus(`error: ${ev.message}`); finish();
      }
    };
    activeStream.onerror = () => { onStatus("stream closed"); finish(); };
  }

  function finish() { closeStream(); activeJob = null; applyState("idle"); }

  async function cancelJob() {
    if (!activeJob) return;
    const id = activeJob; activeJob = null; closeStream();
    try { await fetch(`/api/projections/${id}`, { method: "DELETE" }); } catch { /* ignore */ }
  }

  async function onPause() {
    if (state === "running") { await cancelJob(); applyState("paused"); onStatus("paused"); }
    else if (state === "paused" && lastSubmit) { await submit(lastSubmit); }
  }

  async function onRerun() {
    if (state === "running") await cancelJob();
    await submit(buildBody());
  }

  function mount() {
    tabs.forEach((t) => t.addEventListener("click", () => { if (!t.disabled) selectTab(t.dataset.tab); }));
    runBtn.addEventListener("click", () => submit(buildBody()));
    pauseBtn.addEventListener("click", onPause);
    rerunBtn.addEventListener("click", onRerun);
    selectTab("pca");
    setIter(0, 0);
    applyState("idle");
  }

  return { mount };
}
