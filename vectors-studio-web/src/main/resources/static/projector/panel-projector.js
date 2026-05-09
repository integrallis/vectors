// Right-rail Projector panel. Iter 1 ships the PCA tab only; t-SNE and UMAP
// arrive in Iter 2. POSTs to /api/projections, subscribes to the SSE event
// stream, and forwards coordinates to the scene through onPoints().

const ALGOS = {
  pca: "PCA",
  tsne: "TSNE",
  umap: "UMAP",
};

export function createProjectorPanel({ root, collection, onPoints, onStatus }) {
  const tabs = root.querySelectorAll(".projector-tab");
  const tabPanels = root.querySelectorAll(".projector-tab-panel");
  const dimRadios = root.querySelectorAll('input[name="proj-dim"]');
  const runBtn = root.querySelector("#proj-run");
  const rerunBtn = root.querySelector("#proj-rerun");
  const cancelBtn = root.querySelector("#proj-cancel");
  const iterEl = root.querySelector("#proj-iter");

  let activeTab = "pca";
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
    return {};
  }

  function setIter(i, total) {
    iterEl.textContent = `iter ${i ?? 0} / ${total ?? 0}`;
  }

  function closeStream() {
    if (activeStream) {
      activeStream.close();
      activeStream = null;
    }
  }

  async function run() {
    closeStream();
    const dim = selectedDim();
    const algorithm = ALGOS[activeTab];
    const body = {
      collection,
      algorithm,
      dimensions: dim,
      sampleSize: 0,
      params: paramsFor(activeTab),
      sphereize: false,
    };
    lastSubmit = body;
    runBtn.disabled = true;
    rerunBtn.disabled = true;
    cancelBtn.disabled = false;
    onStatus(`submitting ${algorithm}…`);
    setIter(0, 0);
    let resp;
    try {
      resp = await fetch("/api/projections", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) {
      onStatus(`network error: ${e.message}`);
      runBtn.disabled = false;
      cancelBtn.disabled = true;
      return;
    }
    if (!resp.ok) {
      onStatus(`submit failed: ${resp.status}`);
      runBtn.disabled = false;
      cancelBtn.disabled = true;
      return;
    }
    const { jobId, n } = await resp.json();
    activeJob = jobId;
    onStatus(`running ${algorithm} on ${n} points…`);
    activeStream = new EventSource(`/api/projections/${jobId}/events`);
    activeStream.onmessage = (e) => {
      let ev;
      try {
        ev = JSON.parse(e.data);
      } catch {
        return;
      }
      if (ev.coords) {
        onPoints(ev.coords, dim);
        setIter(ev.iter, ev.total);
        onStatus(`iter ${ev.iter} / ${ev.total}`);
      } else if (ev.result) {
        onPoints(ev.result.coords, dim);
        onStatus(`done · ${ev.result.durationMs} ms`);
        setIter(ev.result.coords.length ? ev.result.coords[0].length : 0, 0);
        finish();
      } else if (ev.message) {
        onStatus(`error: ${ev.message}`);
        finish();
      }
    };
    activeStream.onerror = () => {
      onStatus("stream closed");
      finish();
    };
  }

  function finish() {
    closeStream();
    activeJob = null;
    runBtn.disabled = false;
    rerunBtn.disabled = lastSubmit == null;
    cancelBtn.disabled = true;
  }

  async function cancel() {
    if (!activeJob) return;
    const id = activeJob;
    closeStream();
    onStatus("cancelling…");
    try {
      await fetch(`/api/projections/${id}`, { method: "DELETE" });
    } catch {
      // ignore
    }
    finish();
    onStatus("cancelled");
  }

  function mount() {
    tabs.forEach((t) =>
      t.addEventListener("click", () => {
        if (t.disabled) return;
        selectTab(t.dataset.tab);
      }),
    );
    runBtn.addEventListener("click", run);
    rerunBtn.addEventListener("click", run);
    cancelBtn.addEventListener("click", cancel);
    selectTab("pca");
    setIter(0, 0);
  }

  return { mount };
}
