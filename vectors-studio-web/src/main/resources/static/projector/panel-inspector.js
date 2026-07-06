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

export function createInspectorPanel({ root, collection, scene, dataPanel, canvasHost, onReset, revealPoint }) {
  const queryEl = root.querySelector("#ins-query");
  const kEl = root.querySelector("#ins-k");
  const searchBtn = root.querySelector("#ins-search");
  const lassoBtn = root.querySelector("#ins-lasso");
  const isolateBtn = root.querySelector("#ins-isolate");
  const clearBtn = root.querySelector("#ins-clear");
  const showAllBtn = root.querySelector("#ins-show-all");
  const hitsEl = root.querySelector("#ins-hits");
  const statusEl = root.querySelector("#ins-status");
  const mmrEl = root.querySelector("#ins-mmr");
  const mmrLambdaEl = root.querySelector("#ins-mmr-lambda");
  const mmrLambdaField = root.querySelector("#ins-mmr-lambda-field");
  const mmrLambdaOut = root.querySelector("#ins-mmr-lambda-out");

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
    showAllBtn.disabled = !hasSel;
    // Clear is always available — it doubles as "reset to initial state".
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
      li.addEventListener("dblclick", (e) => onHitDblClick(e, h));
      hitsEl.appendChild(li);
    }
  }

  function onHitClick(e, h) {
    if (h.index == null) return;
    if (e.shiftKey) {
      // Shift-click still accumulates a multi-selection.
      if (selection.has(h.index)) selection.delete(h.index); else selection.add(h.index);
      scene.setSelection(selection);
      refreshButtons();
      renderHits();
      return;
    }
    // Plain single-click reveals ONLY this row (no re-pivot): focus that one
    // point in the scene and pop its hover-card at its on-screen position.
    selection = new Set([h.index]);
    scene.setSelection(selection);
    revealPoint?.(h.index);
    refreshButtons();
    renderHits();
  }

  function onHitDblClick(e, h) {
    if (h.index == null) return;
    // Double-click a row re-pivots: make this hit the new query.
    selectByIndex(h.index);
  }

  // Posts the given JSON payload to the search API and applies the result to
  // the scene + hit list. queryIndex is the optional pivot index (set when
  // searching by doc id) to render a primary label.
  async function postSearch(payload, queryIndex) {
    statusEl.textContent = `searching…`;
    try {
      const res = await fetch(`/api/collections/${encodeURIComponent(collection)}/search`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) { statusEl.textContent = `error ${res.status}`; return null; }
      const { hits } = await res.json();
      lastHits = hits.map((h) => ({ ...h, index: dataPanel.indexOfId?.(h.id) ?? null }));
      selection = new Set(queryIndex != null ? [queryIndex] : []);
      for (const h of lastHits) if (h.index != null) selection.add(h.index);
      scene.setSelection(selection);
      // Paint the neighbour hits orange and drop a bullseye where the query
      // vector approximately lands.
      scene.setHits?.(new Set(lastHits.map((h) => h.index).filter((i) => i != null)));
      updateQueryMarker();
      applyLabels(queryIndex);
      refreshButtons();
      renderHits();
      statusEl.textContent = hits.length === 0 ? "no results" : `${hits.length} hits`;
      return { queryIndex, hits: lastHits };
    } catch (e) { statusEl.textContent = `network error`; return null; }
  }

  // When MMR is enabled, pass the lambda so the backend over-fetches and diversity-re-ranks the
  // neighbours — visible in the scene as the highlighted hits spreading out instead of clustering.
  function mmrPayload() {
    return mmrEl && mmrEl.checked ? { mmr: parseFloat(mmrLambdaEl.value) } : {};
  }

  async function searchById(id, k) {
    return postSearch({ id, k, ...mmrPayload() }, dataPanel.indexOfId?.(id));
  }

  async function searchByQuery(query, k) {
    return postSearch({ query, k, ...mmrPayload() }, null);
  }

  function applyLabels(primaryIndex) {
    if (!scene.setLabels) return;
    // Only the pivot/query point keeps a persistent label. The K neighbour hits
    // no longer show always-on text — their text appears on hover / single
    // row-select via the hover-card instead.
    const items = [];
    if (primaryIndex != null) {
      const txt = dataPanel.labelAt(primaryIndex) ?? dataPanel.idAt(primaryIndex);
      if (txt != null) items.push({ index: primaryIndex, text: String(txt), primary: true });
    }
    scene.setLabels(items);
  }

  // The query vector isn't itself a projected point, so approximate where it
  // lands as the score-weighted centroid of the hit points' world positions.
  function updateQueryMarker() {
    if (!scene.setQueryMarker || !scene.positionOf) return;
    let wx = 0, wy = 0, wz = 0, sum = 0;
    let ux = 0, uy = 0, uz = 0, n = 0;
    for (const h of lastHits) {
      if (h.index == null) continue;
      const p = scene.positionOf(h.index);
      if (!p) continue;
      ux += p[0]; uy += p[1]; uz += p[2]; n += 1;
      const w = h.score != null && h.score > 0 ? h.score : 0;
      wx += p[0] * w; wy += p[1] * w; wz += p[2] * w; sum += w;
    }
    if (n === 0) { scene.setQueryMarker(null); return; }
    if (sum > 0) scene.setQueryMarker([wx / sum, wy / sum, wz / sum]);
    else scene.setQueryMarker([ux / n, uy / n, uz / n]); // fall back to plain centroid
  }

  async function doSearch() {
    const q = queryEl.value.trim();
    if (!q) { statusEl.textContent = "enter a query"; return; }
    const k = Math.max(1, parseInt(kEl.value, 10) || 10);
    // If the query exactly matches a known doc id, pivot kNN around that point;
    // otherwise fall through to text-based hybrid search.
    const idx = dataPanel.indexOfId?.(q);
    if (idx != null) await searchById(q, k);
    else await searchByQuery(q, k);
  }

  async function selectByIndex(idx) {
    if (idx == null || idx < 0) {
      selection = new Set();
      lastHits = [];
      scene.setSelection(selection);
      scene.setLabels?.([]);
      scene.setHits?.(new Set());
      scene.setQueryMarker?.(null);
      revealPoint?.(-1);
      refreshButtons();
      renderHits();
      statusEl.textContent = "";
      return;
    }
    const id = dataPanel.idAt(idx);
    if (id == null) return;
    queryEl.value = id;
    const k = Math.max(1, parseInt(kEl.value, 10) || 10);
    await searchById(id, k);
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
      // Inspector-local reset.
      selection = new Set(); lastHits = [];
      scene.setSelection(selection); scene.setIsolated(false);
      scene.setLabels?.([]);
      scene.setHits?.(new Set());
      scene.setQueryMarker?.(null);
      revealPoint?.(-1);
      queryEl.value = "";
      if (kEl.defaultValue !== "") kEl.value = kEl.defaultValue;
      if (lassoOn) setLasso(false);
      refreshButtons(); renderHits(); statusEl.textContent = "";
      // Restart auto-rotation and let the host reset the rest of the page.
      scene.setAutoRotate?.(true);
      onReset?.();
    });
    svg.addEventListener("pointerdown", onDown);
    svg.addEventListener("pointermove", onMove);
    svg.addEventListener("pointerup", onUp);
    svg.addEventListener("pointerleave", onUp);
    // MMR toggle + lambda slider: reveal the slider and re-run the current query live so the
    // diversity effect is immediately visible in the projection.
    const rerun = () => { if (queryEl.value.trim()) doSearch(); };
    if (mmrEl) {
      mmrEl.addEventListener("change", () => {
        if (mmrLambdaField) mmrLambdaField.hidden = !mmrEl.checked;
        rerun();
      });
    }
    if (mmrLambdaEl) {
      mmrLambdaEl.addEventListener("input", () => {
        if (mmrLambdaOut) mmrLambdaOut.textContent = parseFloat(mmrLambdaEl.value).toFixed(2);
        if (mmrEl && mmrEl.checked) rerun();
      });
    }
  }

  return { mount, selectByIndex };
}
