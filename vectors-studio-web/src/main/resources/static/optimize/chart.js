// Plain-Canvas line chart: composite objective per trial + running-best envelope.
// No dependencies; ~60 LOC including axis labels.

const PADDING = { left: 48, right: 16, top: 16, bottom: 28 };

export function drawHistoryChart(canvas, trials) {
  if (!canvas || !canvas.getContext) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.width;
  const cssH = canvas.clientHeight || canvas.height;
  if (canvas.width !== cssW * dpr) {
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const valid = trials.filter((t) => t && Number.isFinite(t.objectiveScore));
  if (valid.length === 0) {
    ctx.fillStyle = "#888";
    ctx.font = "13px Inter, system-ui, sans-serif";
    ctx.fillText("No trials yet", PADDING.left, cssH / 2);
    return;
  }

  const scores = valid.map((t) => t.objectiveScore);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const span = Math.max(1e-9, maxScore - minScore);
  const yLo = minScore - 0.05 * span;
  const yHi = maxScore + 0.05 * span;
  const plotW = cssW - PADDING.left - PADDING.right;
  const plotH = cssH - PADDING.top - PADDING.bottom;
  const xAt = (i) => PADDING.left + (valid.length === 1 ? plotW / 2 : (i / (valid.length - 1)) * plotW);
  const yAt = (v) => PADDING.top + (1 - (v - yLo) / (yHi - yLo)) * plotH;

  // Axes
  ctx.strokeStyle = "#d0d0d8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PADDING.left, PADDING.top);
  ctx.lineTo(PADDING.left, PADDING.top + plotH);
  ctx.lineTo(PADDING.left + plotW, PADDING.top + plotH);
  ctx.stroke();
  ctx.fillStyle = "#666";
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillText(yHi.toFixed(3), 4, PADDING.top + 4);
  ctx.fillText(yLo.toFixed(3), 4, PADDING.top + plotH);
  ctx.fillText(`trial 1`, PADDING.left, cssH - 8);
  ctx.textAlign = "right";
  ctx.fillText(`trial ${valid.length}`, cssW - PADDING.right, cssH - 8);
  ctx.textAlign = "start";

  // Per-trial dots (faded)
  ctx.fillStyle = "#5a8dee";
  valid.forEach((t, i) => {
    ctx.beginPath();
    ctx.arc(xAt(i), yAt(t.objectiveScore), 2.5, 0, Math.PI * 2);
    ctx.fill();
  });

  // Running-best line (solid)
  let best = -Infinity;
  ctx.strokeStyle = "#2c5cd5";
  ctx.lineWidth = 2;
  ctx.beginPath();
  valid.forEach((t, i) => {
    if (t.objectiveScore > best) best = t.objectiveScore;
    const x = xAt(i);
    const y = yAt(best);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
