// Vectors Studio — projector island.
// Submits a projection job, subscribes to its SSE stream, and renders the resulting
// 2D/3D point cloud. Rendering uses Three.js loaded from a CDN; this is a deliberately
// minimal hand-written ES module (no bundler) to keep the build self-contained.
import * as THREE from "https://unpkg.com/three@0.161.0/build/three.module.js";

const form = document.getElementById("proj-form");
const status = document.getElementById("projector-status");
const canvas = document.getElementById("projector-canvas");

let renderer, scene, camera, points;

function initScene(dimensions) {
  if (renderer) {
    renderer.dispose();
    canvas.replaceChildren();
  }
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  canvas.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
  camera.position.set(0, 0, dimensions === 3 ? 30 : 20);
  camera.lookAt(0, 0, 0);
}

function setPoints(coords) {
  if (points) scene.remove(points);
  const positions = new Float32Array(coords.length * 3);
  for (let i = 0; i < coords.length; i++) {
    positions[i * 3 + 0] = coords[i][0] || 0;
    positions[i * 3 + 1] = coords[i][1] || 0;
    positions[i * 3 + 2] = coords[i][2] || 0;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({ color: 0x0a66c2, size: 0.15 });
  points = new THREE.Points(geo, mat);
  scene.add(points);
  renderer.render(scene, camera);
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const fd = new FormData(form);
  const dimensions = parseInt(fd.get("dimensions"), 10);
  const body = {
    collection: fd.get("collection"),
    algorithm: fd.get("algorithm"),
    dimensions,
    sampleSize: parseInt(fd.get("sampleSize"), 10),
  };
  status.textContent = "submitting…";
  initScene(dimensions);
  const res = await fetch("/api/projections", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    status.textContent = "submit failed: " + res.status;
    return;
  }
  const { jobId } = await res.json();
  status.textContent = "running… (job " + jobId + ")";
  const sse = new EventSource("/api/projections/" + jobId + "/events");
  sse.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.coords) {
      setPoints(ev.coords);
      status.textContent = "iter " + ev.iter + "/" + ev.total;
    } else if (ev.result) {
      setPoints(ev.result.coords);
      status.textContent = "done · " + ev.result.durationMs + "ms";
      sse.close();
    } else if (ev.message) {
      status.textContent = "error: " + ev.message;
      sse.close();
    }
  };
  sse.onerror = () => sse.close();
});
