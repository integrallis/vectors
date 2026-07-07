// Vectors Studio — projector page entry point.
// Mounts the 3D scene and wires the right-rail panels. The panel structure and
// control labels mirror the TensorFlow Embedding Projector; see
// docs/projector-native plan for the porting decisions.
import { createScene } from "./scene.js";
import { createProjectorPanel } from "./panel-projector.js";
import { createDataPanel } from "./panel-data.js";
import { createInspectorPanel } from "./panel-inspector.js";
import { colorsForColumn } from "./colors.js";

const shell = document.querySelector(".projector-shell");
if (!shell) {
  throw new Error("projector: .projector-shell not found");
}

const collection = shell.dataset.collection;
const canvasHost = shell.querySelector(".projector-canvas");
const statusEl = shell.querySelector(".projector-status");
// shadcn-style hover-card: an id (muted, small) stacked above the point's text.
const hoverEl = document.createElement("div");
hoverEl.className = "projector-hovercard";
const hoverIdEl = document.createElement("div");
hoverIdEl.className = "projector-hovercard-id";
const hoverTextEl = document.createElement("div");
hoverTextEl.className = "projector-hovercard-text";
hoverEl.appendChild(hoverIdEl);
hoverEl.appendChild(hoverTextEl);
canvasHost.appendChild(hoverEl);

const setStatus = (msg) => { statusEl.textContent = msg; };

let lastDim = 3;

// Show the hover-card for point `idx` anchored near client coords (x, y).
// idx < 0 hides it. idx === -2 is the sentinel for the projected query point:
// show the raw query text (from the scene) instead of a dataset row.
function showHoverCard(idx, x, y) {
  if (idx === -2) {
    const q = scene.queryText || "";
    hoverIdEl.textContent = "query";
    hoverIdEl.classList.add("is-query");
    hoverTextEl.textContent = q;
    hoverTextEl.style.display = q ? "block" : "none";
    hoverTextEl.classList.add("is-query");
    const rect = canvasHost.getBoundingClientRect();
    hoverEl.style.left = `${x - rect.left + 12}px`;
    hoverEl.style.top = `${y - rect.top + 12}px`;
    hoverEl.style.display = "block";
    return;
  }
  hoverIdEl.classList.remove("is-query");
  hoverTextEl.classList.remove("is-query");
  if (idx < 0) { hoverEl.style.display = "none"; return; }
  const id = dataPanel.idAt(idx);
  if (id == null) { hoverEl.style.display = "none"; return; }
  const label = dataPanel.labelAt(idx);
  hoverIdEl.textContent = id;
  if (label != null && label !== "") {
    hoverTextEl.textContent = label;
    hoverTextEl.style.display = "block";
  } else {
    hoverTextEl.textContent = "";
    hoverTextEl.style.display = "none";
  }
  const rect = canvasHost.getBoundingClientRect();
  hoverEl.style.left = `${x - rect.left + 12}px`;
  hoverEl.style.top = `${y - rect.top + 12}px`;
  hoverEl.style.display = "block";
}

function onPointClick(idx) {
  // -1 means click on empty space — clear selection + labels.
  inspectorPanel?.selectByIndex(idx < 0 ? null : idx);
}

const scene = createScene(canvasHost, { onHover: showHoverCard, onClick: onPointClick });

function applyColors() {
  const values = dataPanel.columnValues();
  if (!values) { scene.setColors(null); return; }
  scene.setColors(colorsForColumn(values));
}

const dataPanel = createDataPanel({
  root: shell.querySelector("#panel-data"),
  collection,
  onChange: ({ reason }) => {
    if (reason === "color" || reason === "loaded") applyColors();
    if (reason === "sphereize") projectorPanel.schedule();
  },
});

const projectorPanel = createProjectorPanel({
  root: shell.querySelector("#panel-projector"),
  collection,
  onPoints: (coords, dim) => {
    lastDim = dim;
    scene.setPositions(coords, dim);
    applyColors();
  },
  onStatus: setStatus,
  getSphereize: () => dataPanel.getSphereize(),
});

const inspectorPanel = createInspectorPanel({
  root: shell.querySelector("#panel-inspector"),
  collection,
  scene,
  dataPanel,
  canvasHost,
  // Reveal a single point (row single-click): pop its hover-card at the point's
  // on-screen position. idx < 0 hides the card.
  revealPoint: (idx) => {
    if (idx == null || idx < 0) { showHoverCard(-1); return; }
    const sp = scene.screenPositionOf(idx);
    if (sp) showHoverCard(idx, sp.x, sp.y);
    else showHoverCard(-1);
  },
  onReset: () => {
    dataPanel.reset();
    applyColors();
    projectorPanel.reset();
  },
});

dataPanel.mount();
projectorPanel.mount();
inspectorPanel.mount();
setStatus("idle · " + (shell.dataset.size ?? "?") + " docs");
projectorPanel.run();
