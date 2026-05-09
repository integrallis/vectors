// Color scales for the projector. The "by column" selector in the Data panel
// either picks the constant brand color (no column), a categorical palette
// (string / boolean / tag values), or a continuous turbo-like ramp (numeric
// columns). Output is rgb in 0..1 to match Three.js BufferAttribute("color").

const BRAND = [0x12 / 255, 0x6e / 255, 0x22 / 255];

// Tableau-10 palette, kept hand-tuned for sufficient contrast on light bg.
const CATEGORICAL = [
  [0x4e, 0x79, 0xa7], [0xf2, 0x8e, 0x2b], [0xe1, 0x57, 0x59],
  [0x76, 0xb7, 0xb2], [0x59, 0xa1, 0x4f], [0xed, 0xc9, 0x48],
  [0xaf, 0x7a, 0xa1], [0xff, 0x9d, 0xa7], [0x9c, 0x75, 0x5f],
  [0xba, 0xb0, 0xac],
].map((rgb) => rgb.map((c) => c / 255));

// Five-stop ramp approximating the turbo colormap; good enough at this density.
const RAMP = [
  [0.18, 0.07, 0.37], [0.27, 0.49, 0.74], [0.36, 0.78, 0.40],
  [0.96, 0.78, 0.20], [0.85, 0.16, 0.18],
];

function rampAt(t) {
  if (!Number.isFinite(t)) return BRAND;
  const x = Math.min(1, Math.max(0, t));
  const i = Math.min(RAMP.length - 2, Math.floor(x * (RAMP.length - 1)));
  const f = x * (RAMP.length - 1) - i;
  const a = RAMP[i], b = RAMP[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function isNumericColumn(values) {
  let nonEmpty = 0, numeric = 0;
  for (const v of values) {
    if (v == null || v === "") continue;
    nonEmpty++;
    if (Number.isFinite(parseFloat(v))) numeric++;
  }
  return nonEmpty > 0 && numeric / nonEmpty >= 0.9;
}

/**
 * Builds a per-row RGB scale for the given column values.
 *
 * @param {string[]|null} values column values, parallel to the points array;
 *        pass null/empty for the "no column" identity scale.
 * @returns {(value: string, index: number) => [number, number, number]}
 */
export function buildColorScale(values) {
  if (!values || !values.length) return () => BRAND;
  if (isNumericColumn(values)) {
    let min = Infinity, max = -Infinity;
    for (const v of values) {
      const n = parseFloat(v);
      if (Number.isFinite(n)) { if (n < min) min = n; if (n > max) max = n; }
    }
    const span = max - min;
    return (value) => {
      const n = parseFloat(value);
      if (!Number.isFinite(n)) return BRAND;
      return rampAt(span > 0 ? (n - min) / span : 0.5);
    };
  }
  // Categorical: assign palette slots in the order distinct values appear.
  const slots = new Map();
  for (const v of values) {
    const key = v == null ? "" : String(v);
    if (!slots.has(key)) slots.set(key, slots.size % CATEGORICAL.length);
  }
  return (value) => {
    const key = value == null ? "" : String(value);
    if (!slots.has(key)) return BRAND;
    return CATEGORICAL[slots.get(key)];
  };
}

/** Flattens a column of row values into a Three.js color BufferAttribute payload. */
export function colorsForColumn(values) {
  const scale = buildColorScale(values);
  const out = new Float32Array((values?.length || 0) * 3);
  for (let i = 0; i < (values?.length || 0); i++) {
    const c = scale(values[i], i);
    out[i * 3 + 0] = c[0];
    out[i * 3 + 1] = c[1];
    out[i * 3 + 2] = c[2];
  }
  return out;
}
