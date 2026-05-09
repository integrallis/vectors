// Vectors Studio — optimize design form. Submits the study config as JSON to the API and
// redirects to the progress page on success.
const shell = document.querySelector(".optimize-shell");
const form = document.getElementById("study-form");
if (shell && form) {
  // Toggle the visual "selected" state on radio choices for clearer affordance.
  const choices = form.querySelectorAll(".sampler-choices .choice");
  const refreshChoices = () => {
    choices.forEach((c) => {
      const input = c.querySelector("input");
      if (input.checked) c.classList.add("selected");
      else c.classList.remove("selected");
    });
  };
  choices.forEach((c) => c.addEventListener("click", refreshChoices));
  refreshChoices();

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    const dto = {
      collection: fd.get("collection"),
      sampler: fd.get("sampler"),
      nTrials: Number(fd.get("nTrials")),
      kForMetrics: Number(fd.get("kForMetrics")),
      seed: Number(fd.get("seed")),
      querySampleSize: Number(fd.get("querySampleSize")),
      metadataField: (fd.get("metadataField") || "").trim() || null,
      mMin: Number(fd.get("mMin")),
      mMax: Number(fd.get("mMax")),
      efMin: Number(fd.get("efMin")),
      efMax: Number(fd.get("efMax")),
      recallWeight: Number(fd.get("recallWeight")),
      ndcgWeight: Number(fd.get("ndcgWeight")),
      latencyWeight: Number(fd.get("latencyWeight")),
    };
    const submit = form.querySelector("button[type=submit]");
    submit.disabled = true;
    submit.textContent = "Starting…";
    try {
      const res = await fetch("/api/optimize/studies", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(dto),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const body = await res.json();
      const studyId = body.studyId;
      const collection = shell.dataset.collection;
      window.location.href = `/collections/${collection}/optimize/progress/${studyId}`;
    } catch (err) {
      submit.disabled = false;
      submit.textContent = "Start study";
      const banner = document.createElement("div");
      banner.className = "form-error";
      banner.textContent = `Could not start study: ${err.message}`;
      form.prepend(banner);
    }
  });
}
