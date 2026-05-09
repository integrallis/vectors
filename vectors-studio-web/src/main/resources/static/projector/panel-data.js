// Right-rail Data panel. Loads metadata.tsv once, populates the Color by /
// Label by selects from its columns, owns the Sphereize toggle, and exposes
// per-point lookups for the rest of the UI (color/label hover).

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
  const colorSel = root.querySelector("#data-color-by");
  const labelSel = root.querySelector("#data-label-by");
  const editSel = root.querySelector("#data-edit-by");
  const sphereizeEl = root.querySelector("#data-sphereize");

  let dataset = { columns: [], rows: [], byId: new Map(), idToIndex: new Map() };
  let colorBy = NONE;
  let labelBy = NONE;
  let sphereize = false;

  function fillSelect(sel, value) {
    sel.innerHTML = "";
    const optNone = document.createElement("option");
    optNone.value = NONE; optNone.textContent = NONE;
    sel.appendChild(optNone);
    for (const col of dataset.columns) {
      if (col === "id") continue;
      const o = document.createElement("option");
      o.value = col; o.textContent = col;
      sel.appendChild(o);
    }
    sel.value = dataset.columns.includes(value) || value === NONE ? value : NONE;
  }

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
    const enabled = dataset.columns.length > 1;
    fillSelect(colorSel, colorBy);
    fillSelect(labelSel, labelBy);
    if (editSel) {
      fillSelect(editSel, NONE);
      editSel.disabled = !enabled;
    }
    colorSel.disabled = !enabled;
    labelSel.disabled = !enabled;
    sphereizeEl.disabled = false;
    // Default Label by to "text" when present so hovers are useful out of the box.
    if (labelBy === NONE && dataset.columns.includes("text")) {
      labelBy = "text";
      labelSel.value = "text";
    }
    emit("loaded");
  }

  function mount() {
    colorSel.addEventListener("change", () => { colorBy = colorSel.value || NONE; emit("color"); });
    labelSel.addEventListener("change", () => { labelBy = labelSel.value || NONE; emit("label"); });
    sphereizeEl.addEventListener("change", () => { sphereize = !!sphereizeEl.checked; emit("sphereize"); });
    load();
  }

  return {
    mount,
    getColorBy: () => colorBy,
    getLabelBy: () => labelBy,
    getSphereize: () => sphereize,
    columnValues: (col) => values(col ?? colorBy),
    labelAt: (i) => valueAt(i, labelBy),
    idAt,
    indexOfId: (id) => dataset.idToIndex.get(id),
  };
}
