// Trial-table helpers: row rendering + click-to-sort. Operates directly on the DOM so the
// progress page can append rows as SSE events arrive without re-rendering the whole table.

const FMT = {
  num: (n, digits = 4) => (Number.isFinite(n) ? Number(n).toFixed(digits) : "—"),
  int: (n) => (Number.isFinite(n) ? String(Math.round(n)) : "—"),
};

const colSelectors = {
  index: 0,
  trialId: 1,
  m: 2,
  ef: 3,
  recall: 4,
  ndcg: 5,
  latency: 6,
  build: 7,
  objective: 8,
};

export function renderTrialRow(index, result, collection, studyId) {
  const tr = document.createElement("tr");
  tr.dataset.index = String(index);
  const params = result.trial?.params ?? {};
  const cells = [
    String(index + 1),
    result.trial?.trialId ?? "—",
    FMT.int(params.m),
    FMT.int(params.efConstruction),
    FMT.num(result.recallAtK, 3),
    FMT.num(result.ndcgAtK, 3),
    FMT.int(result.latencyP95Us),
    FMT.int(result.buildTimeMs),
    FMT.num(result.objectiveScore),
  ];
  cells.forEach((v, idx) => {
    const td = document.createElement("td");
    td.textContent = v;
    if (idx === 0 || idx === 8) td.classList.add("num-strong");
    tr.appendChild(td);
  });
  const apply = document.createElement("td");
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn btn-link btn-sm";
  btn.textContent = "Apply";
  btn.addEventListener("click", () => applyTrial(collection, studyId, result.trial?.trialId));
  apply.appendChild(btn);
  tr.appendChild(apply);
  return tr;
}

async function applyTrial(collection, studyId, trialId) {
  if (!trialId) return;
  if (!confirm("Capture these trial parameters? This is logged but does not yet rebuild the live index.")) return;
  const res = await fetch(
    `/collections/${collection}/optimize/apply/${studyId}/${trialId}?confirm=true`,
    { method: "POST" }
  );
  alert(await res.text());
}

const sortState = { col: null, dir: 1 };

export function sortTrialTable(th) {
  const col = th.dataset.sort;
  if (sortState.col === col) sortState.dir = -sortState.dir;
  else {
    sortState.col = col;
    sortState.dir = 1;
  }
  const idx = colSelectors[col];
  if (idx == null) return;
  const tbody = document.getElementById("trial-rows");
  const rows = Array.from(tbody.querySelectorAll("tr"));
  rows.sort((a, b) => {
    const av = a.children[idx].textContent;
    const bv = b.children[idx].textContent;
    const an = Number(av);
    const bn = Number(bv);
    if (Number.isFinite(an) && Number.isFinite(bn)) return (an - bn) * sortState.dir;
    return av.localeCompare(bv) * sortState.dir;
  });
  rows.forEach((r) => tbody.appendChild(r));
  document.querySelectorAll("#trial-table th[data-sort]").forEach((h) => {
    h.classList.remove("sorted-asc", "sorted-desc");
  });
  th.classList.add(sortState.dir > 0 ? "sorted-asc" : "sorted-desc");
}
