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
const tooltipEl = document.createElement("div");
tooltipEl.className = "projector-tooltip";
canvasHost.appendChild(tooltipEl);

const setStatus = (msg) => { statusEl.textContent = msg; };

let lastDim = 3;

function showTooltip(idx, x, y) {
  if (idx < 0) { tooltipEl.style.display = "none"; return; }
  const id = dataPanel.idAt(idx);
  const label = dataPanel.labelAt(idx);
  if (id == null) { tooltipEl.style.display = "none"; return; }
  const display = label != null && label !== "" ? `${id} · ${label}` : id;
  tooltipEl.textContent = display;
  const rect = canvasHost.getBoundingClientRect();
  tooltipEl.style.left = `${x - rect.left + 12}px`;
  tooltipEl.style.top = `${y - rect.top + 12}px`;
  tooltipEl.style.display = "block";
}

function onPointClick(idx) {
  // -1 means click on empty space — clear selection + labels.
  inspectorPanel?.selectByIndex(idx < 0 ? null : idx);
}

const scene = createScene(canvasHost, { onHover: showTooltip, onClick: onPointClick });

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
