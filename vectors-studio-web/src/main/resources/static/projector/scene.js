// Three.js scene used by the native projector. Owns the renderer, camera,
// orbit controls, and the single Points object that holds every projected
// vector. setPositions() rebuilds the geometry whenever a new projection
// completes; in 2D mode rotation is locked and z is forced to 0.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BG = 0xfcfcfd;
const POINT_COLOR = 0x126e22;
const POINT_SIZE_2D = 4;   // screen pixels
const POINT_SIZE_3D = 5;   // screen pixels

// Crisp solid disc — small filled circle with a 1px anti-aliased rim. Matches
// the tight pin-prick dots the TF Embedding Projector renders.
function makePointSprite() {
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const g = c.getContext("2d");
  g.fillStyle = "white";
  g.beginPath();
  g.arc(16, 16, 14, 0, Math.PI * 2);
  g.closePath();
  g.fill();
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  return tex;
}

// Faint axis lines through the origin (R/G/B for X/Y/Z) — match the TF
// Embedding Projector's reference axes so the user can orient the cloud.
function makeAxes() {
  const g = new THREE.BufferGeometry();
  const positions = new Float32Array([
    -1, 0, 0, 1, 0, 0,
     0, -1, 0, 0, 1, 0,
     0, 0, -1, 0, 0, 1,
  ]);
  const colors = new Float32Array([
    0.78, 0.18, 0.18, 0.78, 0.18, 0.18,
    0.18, 0.55, 0.22, 0.18, 0.55, 0.22,
    0.20, 0.35, 0.78, 0.20, 0.35, 0.78,
  ]);
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const m = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.7, depthWrite: false,
  });
  return new THREE.LineSegments(g, m);
}

export function createScene(host, { onHover, onClick } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(BG, 1);
  host.appendChild(renderer.domElement);

  // Label overlay — absolutely-positioned divs synced to projected point
  // coordinates each frame. Sits above the canvas, transparent to pointers.
  const labelHost = document.createElement("div");
  labelHost.className = "projector-labels";
  host.appendChild(labelHost);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);

  const axes = makeAxes();
  axes.visible = false;
  scene.add(axes);

  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
  camera.position.set(0, 0, 3);
  camera.lookAt(0, 0, 0);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotateSpeed = 2.4;
  // First user interaction stops the spin permanently (matches TF Projector).
  controls.addEventListener("start", () => {
    if (autoRotateOn) { autoRotateOn = false; applyAutoRotate(); }
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(0), 3),
  );
  const sprite = makePointSprite();
  const material = new THREE.PointsMaterial({
    color: POINT_COLOR,
    size: POINT_SIZE_3D,
    sizeAttenuation: false,
    map: sprite,
    transparent: true,
    alphaTest: 0.5,
    depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  scene.add(points);

  const raycaster = new THREE.Raycaster();
  // Threshold is in world units; recomputed after each setPositions based on
  // the bounding-sphere radius so hover works regardless of projection scale.
  raycaster.params.Points = { threshold: 0.02 };
  const ndc = new THREE.Vector2();
  let hoverIndex = -1;
  let pointCount = 0;
  function onPointerMove(ev) {
    if (!onHover || !pointCount) return;
    const rect = renderer.domElement.getBoundingClientRect();
    ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObject(points, false);
    const idx = hits.length ? hits[0].index : -1;
    if (idx !== hoverIndex) {
      hoverIndex = idx;
      onHover(idx, ev.clientX, ev.clientY);
    } else if (idx !== -1) {
      onHover(idx, ev.clientX, ev.clientY);
    }
  }
  function onPointerLeave() {
    if (!onHover) return;
    if (hoverIndex !== -1) { hoverIndex = -1; onHover(-1, 0, 0); }
  }
  // Double-click selects a point: ray-pick under the cursor and forward the
  // index (or -1 for empty space) to the host. Single clicks orbit/drag only.
  function pickAt(ev) {
    const rect = renderer.domElement.getBoundingClientRect();
    ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObject(points, false);
    return hits.length ? hits[0].index : -1;
  }
  function onDblClick(ev) { if (onClick) onClick(pickAt(ev)); }
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerleave", onPointerLeave);
  renderer.domElement.addEventListener("dblclick", onDblClick);

  function resize() {
    const w = host.clientWidth || 1;
    const h = host.clientHeight || 1;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(host);

  let running = true;
  function animate() {
    if (!running) return;
    controls.update();
    renderer.render(scene, camera);
    syncLabels();
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  function frameTo(arr) {
    if (!arr.length) return { cx: 0, cy: 0, cz: 0, r: 1 };
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < arr.length; i += 3) {
      cx += arr[i]; cy += arr[i + 1]; cz += arr[i + 2];
    }
    const n = arr.length / 3;
    cx /= n; cy /= n; cz /= n;
    let r = 0;
    for (let i = 0; i < arr.length; i += 3) {
      const dx = arr[i] - cx, dy = arr[i + 1] - cy, dz = arr[i + 2] - cz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > r) r = d2;
    }
    r = Math.sqrt(r) || 1;
    const dist = Math.max(r * 2.2, 1);
    controls.target.set(cx, cy, cz);
    // Oblique 3/4 camera so all three axes are visible from the start (TF Projector
    // also drops the camera in at an angle rather than head-on along Z).
    camera.position.set(cx + dist * 0.55, cy + dist * 0.45, cz + dist * 0.75);
    camera.lookAt(cx, cy, cz);
    controls.update();
    return { cx, cy, cz, r };
  }

  let positions = new Float32Array(0);
  let baseColors = null;            // pristine column-derived colors
  let selection = new Set();        // indices currently selected
  let isolated = false;             // when true, only selected points render
  let is2D = false;                 // tracks current projection dimensionality
  let autoRotateOn = true;          // user-facing toggle, gated by is2D + selection
  const DIM_FACTOR = 0.18;          // non-selected color blend toward bg

  // index → { el: HTMLDivElement, primary: boolean }
  const labelEls = new Map();

  function applyAutoRotate() {
    controls.autoRotate = autoRotateOn && !is2D && selection.size === 0;
  }

  function setPositions(coords, dim) {
    const n = coords.length;
    const flat = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const c = coords[i];
      flat[i * 3 + 0] = c[0] || 0;
      flat[i * 3 + 1] = c[1] || 0;
      flat[i * 3 + 2] = dim === 3 ? (c[2] || 0) : 0;
    }
    geometry.setAttribute("position", new THREE.BufferAttribute(flat, 3));
    geometry.computeBoundingSphere();
    positions = flat;
    pointCount = n;
    is2D = dim !== 3;
    material.size = is2D ? POINT_SIZE_2D : POINT_SIZE_3D;
    raycaster.params.Points.threshold = (geometry.boundingSphere?.radius || 1) * 0.02;
    controls.enableRotate = !is2D;
    applyAutoRotate();
    applyDisplay();
    const { cx, cy, cz, r } = frameTo(flat);
    // Anchor the axes at the cloud centroid so they sit inside the data and
    // co-rotate with it under autoRotate / OrbitControls.
    axes.position.set(cx, cy, cz);
    axes.scale.setScalar(r * 1.2);
    axes.visible = !is2D;
  }

  function setColors(colors) {
    baseColors = colors && colors.length === pointCount * 3 ? colors : null;
    applyDisplay();
  }

  function applyDisplay() {
    if (!pointCount) return;
    if (baseColors) {
      const out = new Float32Array(pointCount * 3);
      const hasSel = selection.size > 0;
      const bg = ((BG >> 16) & 0xff) / 255;
      const bgg = ((BG >> 8) & 0xff) / 255;
      const bgb = (BG & 0xff) / 255;
      for (let i = 0; i < pointCount; i++) {
        const sel = selection.has(i);
        const dim = hasSel && !sel;
        const f = dim ? DIM_FACTOR : 1.0;
        out[i * 3 + 0] = baseColors[i * 3 + 0] * f + bg * (1 - f);
        out[i * 3 + 1] = baseColors[i * 3 + 1] * f + bgg * (1 - f);
        out[i * 3 + 2] = baseColors[i * 3 + 2] * f + bgb * (1 - f);
      }
      geometry.setAttribute("color", new THREE.BufferAttribute(out, 3));
      material.vertexColors = true;
    } else {
      geometry.deleteAttribute("color");
      material.vertexColors = false;
    }
    if (isolated && selection.size > 0) {
      const idx = new Uint32Array(selection.size);
      let k = 0;
      for (const i of selection) idx[k++] = i;
      geometry.setIndex(new THREE.BufferAttribute(idx, 1));
    } else {
      geometry.setIndex(null);
    }
    material.needsUpdate = true;
  }

  function setSelection(indices) {
    selection = new Set(indices);
    if (isolated && selection.size === 0) isolated = false;
    applyAutoRotate();
    applyDisplay();
  }

  function setIsolated(v) { isolated = !!v; applyDisplay(); }

  // items: Array<{ index: number, text: string, primary?: boolean }>.
  // Pass [] (or omit) to clear all labels.
  function setLabels(items) {
    const next = new Map();
    for (const it of items || []) {
      if (it == null || it.index == null || it.text == null) continue;
      let entry = labelEls.get(it.index);
      if (!entry) {
        const el = document.createElement("div");
        el.className = "projector-label";
        labelHost.appendChild(el);
        entry = { el, primary: false };
      }
      const primary = !!it.primary;
      if (entry.el.textContent !== it.text) entry.el.textContent = it.text;
      if (entry.primary !== primary) {
        entry.el.classList.toggle("is-primary", primary);
        entry.primary = primary;
      }
      next.set(it.index, entry);
    }
    for (const [idx, entry] of labelEls) {
      if (!next.has(idx)) entry.el.remove();
    }
    labelEls.clear();
    for (const [idx, entry] of next) labelEls.set(idx, entry);
    syncLabels();
  }

  const _v = new THREE.Vector3();
  function syncLabels() {
    if (!labelEls.size) return;
    const w = renderer.domElement.clientWidth;
    const h = renderer.domElement.clientHeight;
    for (const [idx, entry] of labelEls) {
      _v.set(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
      _v.project(camera);
      // Hide labels for points behind the camera (z>1 in NDC after project).
      if (_v.z < -1 || _v.z > 1) { entry.el.style.display = "none"; continue; }
      entry.el.style.display = "block";
      const x = (_v.x * 0.5 + 0.5) * w;
      const y = (-_v.y * 0.5 + 0.5) * h;
      entry.el.style.transform = `translate(${x}px, ${y}px)`;
    }
  }

  function projectAll(targetXY) {
    // Fills targetXY with NDC-ish pixel coords; callers should pass a
    // Float32Array of length pointCount*2.
    const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const v = new THREE.Vector3();
    const w = renderer.domElement.clientWidth;
    const h = renderer.domElement.clientHeight;
    for (let i = 0; i < pointCount; i++) {
      v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).applyMatrix4(m);
      targetXY[i * 2] = (v.x * 0.5 + 0.5) * w;
      targetXY[i * 2 + 1] = (-v.y * 0.5 + 0.5) * h;
    }
  }

  function setControlsEnabled(v) { controls.enabled = !!v; }

  function setAutoRotate(v) { autoRotateOn = !!v; applyAutoRotate(); }

  function dispose() {
    running = false;
    ro.disconnect();
    renderer.domElement.removeEventListener("pointermove", onPointerMove);
    renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
    renderer.domElement.removeEventListener("dblclick", onDblClick);
    controls.dispose();
    geometry.dispose();
    material.dispose();
    axes.geometry.dispose();
    axes.material.dispose();
    renderer.dispose();
    labelHost.remove();
    if (renderer.domElement.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
  }

  return {
    setPositions, setColors, setSelection, setIsolated, setLabels,
    projectAll, setControlsEnabled, setAutoRotate,
    get pointCount() { return pointCount; },
    canvas: renderer.domElement,
    dispose,
  };
}
