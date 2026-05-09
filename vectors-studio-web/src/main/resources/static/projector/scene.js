// Three.js scene used by the native projector. Owns the renderer, camera,
// orbit controls, and the single Points object that holds every projected
// vector. setPositions() rebuilds the geometry whenever a new projection
// completes; in 2D mode rotation is locked and z is forced to 0.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BG = 0xfcfcfd;
const POINT_COLOR = 0x126e22;
const POINT_SIZE_2D = 0.04;
const POINT_SIZE_3D = 0.05;

export function createScene(host, { onHover } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(BG, 1);
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);

  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
  camera.position.set(0, 0, 3);
  camera.lookAt(0, 0, 0);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(0), 3),
  );
  const material = new THREE.PointsMaterial({
    color: POINT_COLOR,
    size: POINT_SIZE_3D,
    sizeAttenuation: true,
  });
  const points = new THREE.Points(geometry, material);
  scene.add(points);

  const raycaster = new THREE.Raycaster();
  raycaster.params.Points = { threshold: POINT_SIZE_3D * 0.6 };
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
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerleave", onPointerLeave);

  function resize() {
    const w = host.clientWidth || 1;
    const h = host.clientHeight || 1;
    renderer.setSize(w, h, false);
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
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  function frameTo(arr) {
    if (!arr.length) return;
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
    r = Math.sqrt(r);
    const dist = Math.max(r * 2.2, 1);
    controls.target.set(cx, cy, cz);
    camera.position.set(cx, cy, cz + dist);
    camera.lookAt(cx, cy, cz);
    controls.update();
  }

  let positions = new Float32Array(0);
  let baseColors = null;            // pristine column-derived colors
  let selection = new Set();        // indices currently selected
  let isolated = false;             // when true, only selected points render
  const DIM_FACTOR = 0.18;          // non-selected color blend toward bg

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
    const is2d = dim !== 3;
    material.size = is2d ? POINT_SIZE_2D : POINT_SIZE_3D;
    raycaster.params.Points.threshold = material.size * 0.6;
    controls.enableRotate = !is2d;
    applyDisplay();
    frameTo(flat);
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
    applyDisplay();
  }

  function setIsolated(v) { isolated = !!v; applyDisplay(); }

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

  function dispose() {
    running = false;
    ro.disconnect();
    renderer.domElement.removeEventListener("pointermove", onPointerMove);
    renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
    controls.dispose();
    geometry.dispose();
    material.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
  }

  return {
    setPositions, setColors, setSelection, setIsolated,
    projectAll, setControlsEnabled,
    get pointCount() { return pointCount; },
    canvas: renderer.domElement,
    dispose,
  };
}
