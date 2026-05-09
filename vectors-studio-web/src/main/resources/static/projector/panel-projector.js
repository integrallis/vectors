// Right-rail Projector panel. Hosts the PCA / t-SNE / UMAP segmented toggle and
// the 2D/3D toggle, and auto-submits a new projection whenever any input
// changes. Server-side parameter keys live in ApiRoutes.paramsFrom.

const ALGOS = { pca: "PCA", tsne: "TSNE", umap: "UMAP" };
const DEBOUNCE_MS = 350;

export function createProjectorPanel({ root, collection, onPoints, onStatus, getSphereize }) {
  const algoBtns = root.querySelectorAll(".seg-toggle [data-algo]");
  const algoPanels = root.querySelectorAll(".projector-tab-panel");
  const dimBtns = root.querySelectorAll(".seg-toggle [data-dim]");

  const num = (sel, dflt) => {
    const el = root.querySelector(sel);
    const v = parseFloat(el?.value);
    return Number.isFinite(v) ? v : dflt;
  };
  const intNum = (sel, dflt) => Math.trunc(num(sel, dflt));

  let activeAlgo = "pca";
  let activeDim = 3;
  let activeJob = null;
  let activeStream = null;
  let pending = null; // setTimeout handle for debounced submit

  function selectAlgo(algo) {
    activeAlgo = algo;
    algoBtns.forEach((b) => b.classList.toggle("is-active", b.dataset.algo === algo));
    algoPanels.forEach((p) => p.classList.toggle("is-active", p.dataset.tab === algo));
  }
  function selectDim(dim) {
    activeDim = dim;
    dimBtns.forEach((b) => b.classList.toggle("is-active", parseInt(b.dataset.dim, 10) === dim));
  }

  function paramsFor(algo) {
    if (algo === "pca") return { center: true, whiten: false };
    if (algo === "tsne") return {
      perplexity: intNum("#tsne-perplexity", 15),
      learningRate: num("#tsne-learning-rate", 200),
      iterations: intNum("#tsne-iterations", 1000),
      seed: intNum("#tsne-seed", 42),
    };
    if (algo === "umap") return {
      neighbors: intNum("#umap-neighbors", 15),
      minDist: num("#umap-min-dist", 0.1),
      iterations: intNum("#umap-iterations", 200),
      seed: intNum("#umap-seed", 42),
    };
    return {};
  }

  function closeStream() {
    if (activeStream) { activeStream.close(); activeStream = null; }
  }

  async function cancelJob() {
    if (!activeJob) return;
    const id = activeJob; activeJob = null; closeStream();
    try { await fetch(`/api/projections/${id}`, { method: "DELETE" }); } catch { /* ignore */ }
  }

  function buildBody() {
    return {
      collection,
      algorithm: ALGOS[activeAlgo],
      dimensions: activeDim,
      sampleSize: 0,
      params: paramsFor(activeAlgo),
      sphereize: getSphereize ? !!getSphereize() : false,
    };
  }

  async function submit(body) {
    await cancelJob();
    onStatus(`submitting ${body.algorithm}…`);
    let resp;
    try {
      resp = await fetch("/api/projections", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) { onStatus(`network error: ${e.message}`); return; }
    if (!resp.ok) { onStatus(`submit failed: ${resp.status}`); return; }
    const { jobId, n } = await resp.json();
    activeJob = jobId;
    onStatus(`running ${body.algorithm} on ${n} points…`);
    activeStream = new EventSource(`/api/projections/${jobId}/events`);
    activeStream.onmessage = (e) => {
      let ev; try { ev = JSON.parse(e.data); } catch { return; }
      const dim = body.dimensions;
      if (ev.coords) {
        onPoints(ev.coords, dim);
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

  function finish() { closeStream(); activeJob = null; }

  function schedule() {
    if (pending) clearTimeout(pending);
    pending = setTimeout(() => { pending = null; submit(buildBody()); }, DEBOUNCE_MS);
  }

  function mount() {
    algoBtns.forEach((b) =>
      b.addEventListener("click", () => { selectAlgo(b.dataset.algo); schedule(); }));
    dimBtns.forEach((b) =>
      b.addEventListener("click", () => { selectDim(parseInt(b.dataset.dim, 10)); schedule(); }));
    // Hyperparameter inputs: debounced auto-rerun on change.
    root.querySelectorAll(".projector-tab-panel input[type=number]").forEach((el) =>
      el.addEventListener("change", schedule));
    selectAlgo("pca");
    selectDim(3);
  }

  function run() { return submit(buildBody()); }

  // Reset algorithm + dim to PCA / 3D and restore numeric input defaults from
  // their HTML defaultValue. Schedules a fresh projection.
  function reset() {
    selectAlgo("pca");
    selectDim(3);
    root.querySelectorAll(".projector-tab-panel input[type=number]").forEach((el) => {
      if (el.defaultValue !== "") el.value = el.defaultValue;
    });
    schedule();
  }

  return { mount, run, schedule, reset };
}
