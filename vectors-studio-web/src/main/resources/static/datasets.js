/*
 * Drives the Sample-datasets tab: fetches per-row loaded status from the catalog, kicks off async
 * loads, and streams progress over SSE into an inline progress bar. Vanilla JS, no build step.
 */
(function () {
  "use strict";

  // model id -> true when some provider serving it has its key present. Populated from
  // /api/providers at init and consulted by the Load confirmation modal.
  var configuredModels = {};
  // dataset id -> catalog entry (carries model + queryModel for the modal copy).
  var catalogById = {};

  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        node.setAttribute(k, attrs[k]);
      });
    }
    if (text != null) {
      node.textContent = text;
    }
    return node;
  }

  function renderLoaded(row) {
    var id = row.dataset.id;
    var status = row.querySelector(".ds-status");
    var actions = row.querySelector(".ds-actions");
    status.replaceChildren(el("span", { class: "pill pill-sample", title: "loaded" }, "loaded"));
    actions.replaceChildren(
      el("a", { class: "btn", href: "/collections/" + id + "/projector" }, "Open")
    );
  }

  function renderIdle(row) {
    var status = row.querySelector(".ds-status");
    var actions = row.querySelector(".ds-actions");
    status.replaceChildren(el("span", { class: "muted" }, "not loaded"));
    var btn = el("button", { class: "btn btn-primary", type: "button" }, "Load");
    btn.addEventListener("click", function () {
      openLoadModal(row, btn);
    });
    actions.replaceChildren(btn);
  }

  function renderProgress(row) {
    var status = row.querySelector(".ds-status");
    var total = parseInt(row.dataset.total, 10) || 0;
    var bar = el("progress", { class: "ds-progress", max: String(total), value: "0" });
    var label = el("span", { class: "muted ds-progress-label" }, "0 / " + total);
    status.replaceChildren(bar, label);
    return { bar: bar, label: label };
  }

  function renderError(row, btn, message) {
    var status = row.querySelector(".ds-status");
    status.replaceChildren(el("span", { class: "form-error", title: message }, "error: " + message));
    if (btn) {
      btn.disabled = false;
    }
  }

  // Informational confirmation before a load: reminds the user which model the vectors use and
  // whether a provider is configured to serve the query model needed for text search.
  function openLoadModal(row, btn) {
    var modal = document.getElementById("ds-modal");
    if (!modal) {
      startLoad(row, btn);
      return;
    }
    var d = catalogById[row.dataset.id] || {};
    var configured = !!(d.queryModel && configuredModels[d.queryModel]);
    var msg = document.getElementById("ds-modal-msg");
    msg.replaceChildren();
    msg.appendChild(document.createTextNode("This dataset's vectors are pre-computed with "));
    msg.appendChild(el("strong", null, d.model || "an embedding model"));
    msg.appendChild(document.createTextNode("."));
    if (d.queryModel) {
      msg.appendChild(
        document.createTextNode(" To search it by text you'll also need a provider serving ")
      );
      msg.appendChild(el("strong", null, d.queryModel));
      msg.appendChild(document.createTextNode(" — "));
      if (configured) {
        msg.appendChild(el("span", { class: "pill pill-sample" }, "✓ configured"));
      } else {
        msg.appendChild(el("span", { class: "form-error" }, "✗ not configured (add on Providers)"));
      }
      msg.appendChild(document.createTextNode("."));
    }
    modal.style.display = "flex";
    var loadBtn = document.getElementById("ds-modal-load");
    var cancelBtn = document.getElementById("ds-modal-cancel");
    function close() {
      modal.style.display = "none";
      loadBtn.onclick = null;
      cancelBtn.onclick = null;
    }
    loadBtn.onclick = function () {
      close();
      startLoad(row, btn);
    };
    cancelBtn.onclick = close;
  }

  function startLoad(row, btn) {
    var id = row.dataset.id;
    btn.disabled = true;
    fetch("/api/datasets/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: id })
    })
      .then(function (r) {
        if (!r.ok) {
          return r.text().then(function (t) {
            throw new Error(t || "HTTP " + r.status);
          });
        }
        return r.json();
      })
      .then(function (data) {
        streamProgress(row, btn, data.jobId);
      })
      .catch(function (err) {
        renderError(row, btn, err.message);
      });
  }

  function streamProgress(row, btn, jobId) {
    var ui = renderProgress(row);
    var es = new EventSource("/api/datasets/load/" + jobId + "/events");
    es.onmessage = function (ev) {
      var d;
      try {
        d = JSON.parse(ev.data);
      } catch (e) {
        return;
      }
      if (typeof d.total === "number" && d.total > 0) {
        ui.bar.max = d.total;
      }
      ui.bar.value = d.loaded;
      ui.label.textContent = d.loaded + " / " + d.total;
      if (d.state === "DONE") {
        es.close();
        renderLoaded(row);
      } else if (d.state === "ERROR") {
        es.close();
        renderError(row, btn, d.error || "load failed");
      }
    };
    es.onerror = function () {
      // The stream closes cleanly on terminal events; only surface an error if we never finished.
      if (es.readyState === EventSource.CLOSED && row.querySelector(".ds-progress")) {
        renderError(row, btn, "connection lost");
      }
    };
  }

  function loadProviders() {
    return fetch("/api/providers")
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        (data.providers || []).forEach(function (p) {
          if (p.keyPresent && Array.isArray(p.models)) {
            p.models.forEach(function (m) {
              configuredModels[m] = true;
            });
          }
        });
      })
      .catch(function () {
        // Providers status is advisory for the modal; a fetch failure just shows "not configured".
      });
  }

  function init() {
    loadProviders().then(function () {
      fetch("/api/datasets/catalog")
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          (data.datasets || []).forEach(function (d) {
            catalogById[d.id] = d;
          });
          document.querySelectorAll("tr[data-id]").forEach(function (row) {
            var d = catalogById[row.dataset.id];
          if (!d) {
            return;
          }
          if (typeof d.defaultLimit === "number") {
            row.dataset.total = d.defaultLimit;
          }
          if (d.loaded) {
            renderLoaded(row);
          } else {
            renderIdle(row);
          }
          });
        });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
