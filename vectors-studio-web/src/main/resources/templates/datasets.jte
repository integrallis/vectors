@import java.util.List
@import com.integrallis.vectors.studio.web.dataset.DatasetCatalogEntry
@param List<DatasetCatalogEntry> entries
@template.layout(
    title = "Sample datasets",
    head = @`<script src="/static/datasets.js" defer></script>`,
    body = @`
      <div class="page-head">
        <h1>Sample datasets</h1>
        <a class="btn" href="/collections">Back to collections</a>
      </div>
      <p class="muted">
        Load a curated embedding dataset from the HuggingFace datasets-server straight into a live
        collection, then explore it in the projector.
      </p>
      <table class="collection-table datasets-table">
        <thead>
          <tr>
            <th>Dataset</th><th>Domain</th><th>Model</th><th>Dim</th><th>Status</th>
            <th class="actions-col">Actions</th>
          </tr>
        </thead>
        <tbody>
          @for(DatasetCatalogEntry e : entries)
            <tr data-id="${e.id()}" data-total="${e.defaultLimit()}">
              <td>
                <div class="dataset-name">${e.name()}</div>
                <div class="muted dataset-desc">${e.description()}</div>
              </td>
              <td>
                <span class="pill pill-metric" title="domain">${e.domain()}</span>
              </td>
              <td class="muted">${e.model()}</td>
              <td>
                <span class="pill pill-dim" title="dimension">d=${e.dimension()}</span>
              </td>
              <td class="ds-status"><span class="muted">…</span></td>
              <td class="actions-col ds-actions"></td>
            </tr>
          @endfor
        </tbody>
      </table>
      <div id="ds-modal" class="ds-modal" style="display:none;position:fixed;inset:0;z-index:50;align-items:center;justify-content:center;background:rgba(0,0,0,0.45);">
        <div class="ds-modal-card" style="background:var(--bg,#fff);color:inherit;max-width:32rem;width:calc(100% - 2rem);padding:1.25rem 1.5rem;border-radius:0.5rem;box-shadow:0 10px 40px rgba(0,0,0,0.35);">
          <h2 id="ds-modal-title" style="margin-top:0;">Load dataset</h2>
          <p id="ds-modal-msg" class="muted"></p>
          <div style="display:flex;gap:0.5rem;justify-content:flex-end;margin-top:1rem;">
            <button type="button" class="btn" id="ds-modal-cancel">Cancel</button>
            <button type="button" class="btn btn-primary" id="ds-modal-load">Load</button>
          </div>
        </div>
      </div>
    `
)
