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

export function createScene(host) {
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
    const is2d = dim !== 3;
    material.size = is2d ? POINT_SIZE_2D : POINT_SIZE_3D;
    controls.enableRotate = !is2d;
    frameTo(flat);
  }

  function dispose() {
    running = false;
    ro.disconnect();
    controls.dispose();
    geometry.dispose();
    material.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode) {
      renderer.domElement.parentNode.removeChild(renderer.domElement);
    }
  }

  return { setPositions, dispose };
}
