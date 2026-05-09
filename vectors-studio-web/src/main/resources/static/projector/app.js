// Vectors Studio — projector page entry point.
// Mounts the 3D scene and wires the right-rail panels. The panel structure and
// control labels mirror the TensorFlow Embedding Projector; see
// docs/projector-native plan for the porting decisions.
import { createScene } from "./scene.js";
import { createProjectorPanel } from "./panel-projector.js";

const shell = document.querySelector(".projector-shell");
if (!shell) {
  throw new Error("projector: .projector-shell not found");
}

const collection = shell.dataset.collection;
const canvasHost = shell.querySelector(".projector-canvas");
const statusEl = shell.querySelector(".projector-status");

const setStatus = (msg) => {
  statusEl.textContent = msg;
};

const scene = createScene(canvasHost);

const projectorPanel = createProjectorPanel({
  root: shell.querySelector("#panel-projector"),
  collection,
  onPoints: (coords, dim) => scene.setPositions(coords, dim),
  onStatus: setStatus,
});

projectorPanel.mount();
setStatus("idle · " + (shell.dataset.size ?? "?") + " docs");
