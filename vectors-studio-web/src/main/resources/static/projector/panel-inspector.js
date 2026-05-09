// Inspector panel: search by id (POST /api/collections/{name}/search), render
// the neighbor list, drive selection / isolate / show-all, and host the lasso
// tool. Selection is index-based and round-trips through the scene.

const SVG_NS = "http://www.w3.org/2000/svg";

function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 2; i < poly.length; j = i, i += 2) {
    const xi = poly[i], yi = poly[i + 1];
    const xj = poly[j], yj = poly[j + 1];
    const intersect = ((yi > y) !== (yj > y)) &&
      (x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

export function createInspectorPanel({ root, collection, scene, dataPanel, canvasHost }) {
  const queryEl = root.querySelector("#ins-query");
  const kEl = root.querySelector("#ins-k");
  const searchBtn = root.querySelector("#ins-search");
  const lassoBtn = root.querySelector("#ins-lasso");
  const isolateBtn = root.querySelector("#ins-isolate");
  const clearBtn = root.querySelector("#ins-clear");
  const showAllBtn = root.querySelector("#ins-show-all");
  const hitsEl = root.querySelector("#ins-hits");
  const statusEl = root.querySelector("#ins-status");

  // SVG overlay used by the lasso tool. Lives alongside the WebGL canvas.
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "projector-lasso");
  svg.style.display = "none";
  const path = document.createElementNS(SVG_NS, "path");
  svg.appendChild(path);
  canvasHost.appendChild(svg);

  let lastHits = [];
  let selection = new Set();
  let lassoOn = false;

  function refreshButtons() {
    const hasSel = selection.size > 0;
    isolateBtn.disabled = !hasSel;
    clearBtn.disabled = !hasSel;
    showAllBtn.disabled = !hasSel;
  }

  function renderHits() {
    hitsEl.innerHTML = "";
    for (const h of lastHits) {
      const li = document.createElement("li");
      li.dataset.id = h.id;
      const idx = h.index;
      if (idx != null && selection.has(idx)) li.classList.add("is-selected");
      const left = document.createElement("span");
      left.textContent = h.id;
      const right = document.createElement("span");
      right.textContent = h.score != null ? h.score.toFixed(3) : "";
      li.appendChild(left); li.appendChild(right);
      li.addEventListener("click", (e) => onHitClick(e, h));
      hitsEl.appendChild(li);
    }
  }

  function onHitClick(e, h) {
    if (h.index == null) return;
    if (e.shiftKey) {
      if (selection.has(h.index)) selection.delete(h.index); else selection.add(h.index);
    } else {
      selection = new Set([h.index]);
    }
    scene.setSelection(selection);
    refreshButtons();
    renderHits();
  }

  async function doSearch() {
    const id = queryEl.value.trim();
    if (!id) { statusEl.textContent = "enter an id"; return; }
    const k = Math.max(1, parseInt(kEl.value, 10) || 10);
    statusEl.textContent = `searching…`;
    try {
      const res = await fetch(`/api/collections/${encodeURIComponent(collection)}/search`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ id, k }),
      });
      if (!res.ok) { statusEl.textContent = `error ${res.status}`; return; }
      const { hits } = await res.json();
      lastHits = hits.map((h) => ({ ...h, index: dataPanel.indexOfId?.(h.id) ?? null }));
      const queryIndex = dataPanel.indexOfId?.(id);
      selection = new Set(queryIndex != null ? [queryIndex] : []);
      for (const h of lastHits) if (h.index != null) selection.add(h.index);
      scene.setSelection(selection);
      refreshButtons();
      renderHits();
      statusEl.textContent = `${hits.length} neighbours`;
    } catch (e) { statusEl.textContent = `network error`; }
  }

  function setLasso(on) {
    lassoOn = on;
    lassoBtn.setAttribute("aria-pressed", String(on));
    scene.setControlsEnabled(!on);
    svg.style.pointerEvents = on ? "auto" : "none";
    if (!on) { svg.style.display = "none"; path.removeAttribute("d"); }
  }

  let drawing = false, pts = [];
  function onDown(e) {
    if (!lassoOn) return;
    drawing = true; pts = [];
    const r = svg.getBoundingClientRect();
    pts.push(e.clientX - r.left, e.clientY - r.top);
    svg.style.display = "block";
    path.setAttribute("d", `M${pts[0]} ${pts[1]}`);
  }
  function onMove(e) {
    if (!drawing) return;
    const r = svg.getBoundingClientRect();
    pts.push(e.clientX - r.left, e.clientY - r.top);
    path.setAttribute("d", path.getAttribute("d") + ` L${pts[pts.length - 2]} ${pts[pts.length - 1]}`);
  }
  function onUp() {
    if (!drawing) return;
    drawing = false;
    if (pts.length >= 6 && scene.pointCount > 0) {
      const xy = new Float32Array(scene.pointCount * 2);
      scene.projectAll(xy);
      const sel = new Set();
      for (let i = 0; i < scene.pointCount; i++) {
        if (pointInPolygon(xy[i * 2], xy[i * 2 + 1], pts)) sel.add(i);
      }
      selection = sel;
      scene.setSelection(selection);
      refreshButtons(); renderHits();
      statusEl.textContent = `${selection.size} selected`;
    }
    setLasso(false);
  }

  function mount() {
    searchBtn.addEventListener("click", doSearch);
    queryEl.addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
    lassoBtn.addEventListener("click", () => setLasso(!lassoOn));
    isolateBtn.addEventListener("click", () => scene.setIsolated(true));
    showAllBtn.addEventListener("click", () => scene.setIsolated(false));
    clearBtn.addEventListener("click", () => {
      selection = new Set(); scene.setSelection(selection); scene.setIsolated(false);
      refreshButtons(); renderHits(); statusEl.textContent = "";
    });
    svg.addEventListener("pointerdown", onDown);
    svg.addEventListener("pointermove", onMove);
    svg.addEventListener("pointerup", onUp);
    svg.addEventListener("pointerleave", onUp);
  }

  return { mount };
}
