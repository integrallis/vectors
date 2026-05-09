// Right-rail Data panel. Loads metadata.tsv once, owns the Sphereize toggle,
// and exposes per-point lookups for the rest of the UI (label hover, etc.).
// Label-by defaults to the "text" column when present so tooltips stay useful.

const NONE = "(none)";

function parseTsv(text) {
  const lines = text.split(/\r?\n/);
  while (lines.length && !lines[lines.length - 1]) lines.pop();
  if (!lines.length) return { columns: [], rows: [], byId: new Map(), idToIndex: new Map() };
  const header = lines[0].split("\t");
  const rows = new Array(lines.length - 1);
  const byId = new Map();
  const idToIndex = new Map();
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split("\t");
    const row = {};
    for (let c = 0; c < header.length; c++) row[header[c]] = cells[c] ?? "";
    rows[i - 1] = row;
    byId.set(row.id, row);
    idToIndex.set(row.id, i - 1);
  }
  return { columns: header, rows, byId, idToIndex };
}

export function createDataPanel({ root, collection, onChange }) {
  const sphereizeEl = root.querySelector("#data-sphereize");

  let dataset = { columns: [], rows: [], byId: new Map(), idToIndex: new Map() };
  let colorBy = NONE;
  let labelBy = NONE;
  let sphereize = false;

  function values(column) {
    if (column === NONE || !dataset.rows.length) return null;
    const out = new Array(dataset.rows.length);
    for (let i = 0; i < dataset.rows.length; i++) out[i] = dataset.rows[i][column] ?? "";
    return out;
  }

  function valueAt(index, column) {
    if (index < 0 || index >= dataset.rows.length || column === NONE) return null;
    return dataset.rows[index][column] ?? "";
  }

  function idAt(index) {
    if (index < 0 || index >= dataset.rows.length) return null;
    return dataset.rows[index].id;
  }

  function emit(reason) {
    onChange?.({ reason, colorBy, labelBy, sphereize });
  }

  async function load() {
    try {
      const res = await fetch(`/api/collections/${encodeURIComponent(collection)}/metadata.tsv`);
      if (!res.ok) throw new Error(`metadata fetch failed: ${res.status}`);
      dataset = parseTsv(await res.text());
    } catch (e) {
      dataset = { columns: ["id"], rows: [], byId: new Map(), idToIndex: new Map() };
      console.warn("projector: metadata load failed", e);
    }
    if (sphereizeEl) sphereizeEl.disabled = false;
    // Default Label by to "text" when present so hovers are useful out of the box.
    if (labelBy === NONE && dataset.columns.includes("text")) labelBy = "text";
    emit("loaded");
  }

  function mount() {
    sphereizeEl.addEventListener("change", () => { sphereize = !!sphereizeEl.checked; emit("sphereize"); });
    load();
  }

  // Reset to the page-load defaults without firing onChange — the caller is
  // responsible for re-running projection / colors after a coordinated reset.
  function reset() {
    colorBy = NONE;
    labelBy = dataset.columns.includes("text") ? "text" : NONE;
    sphereize = false;
    if (sphereizeEl) sphereizeEl.checked = false;
  }

  return {
    mount,
    reset,
    getColorBy: () => colorBy,
    getLabelBy: () => labelBy,
    getSphereize: () => sphereize,
    columnValues: (col) => values(col ?? colorBy),
    labelAt: (i) => valueAt(i, labelBy),
    idAt,
    indexOfId: (id) => dataset.idToIndex.get(id),
  };
}
