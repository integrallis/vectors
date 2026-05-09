@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param CollectionSummary summary
@template.layout(
    title = "Projector — " + summary.name(),
    head = @`
      <script type="importmap">
        {
          "imports": {
            "three": "https://unpkg.com/three@0.163.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.163.0/examples/jsm/"
          }
        }
      </script>
      <script type="module" src="/static/projector/app.js" defer></script>
    `,
    body = @`
      <section class="projector-shell"
               data-collection="${summary.name()}"
               data-size="${summary.size()}"
               data-dim="${summary.dimension()}">
        <div class="projector-viewport">
          <div class="projector-canvas"></div>
          <div class="projector-status">idle</div>
        </div>
        <aside class="projector-rail">
          <section class="projector-panel" id="panel-data">
            <header class="projector-panel-title">Data</header>
            <div class="projector-panel-body">
              <label class="projector-field">
                <span>Tensor</span>
                <select disabled><option>${summary.name()}</option></select>
              </label>
              <label class="projector-checkbox">
                <input type="checkbox" id="data-sphereize" disabled>
                <span>Sphereize data</span>
              </label>
              <label class="projector-field">
                <span>Color by</span>
                <select id="data-color-by" disabled><option>(none)</option></select>
              </label>
              <label class="projector-field">
                <span>Label by</span>
                <select id="data-label-by" disabled><option>(none)</option></select>
              </label>
              <label class="projector-field">
                <span>Edit by</span>
                <select id="data-edit-by" disabled><option>(none)</option></select>
              </label>
            </div>
          </section>
          <section class="projector-panel" id="panel-projector">
            <header class="projector-panel-title">Projector</header>
            <div class="projector-panel-body">
              <nav class="projector-tabs" role="tablist">
                <button class="projector-tab is-active" data-tab="pca" role="tab">PCA</button>
                <button class="projector-tab" data-tab="tsne" role="tab" disabled>t-SNE</button>
                <button class="projector-tab" data-tab="umap" role="tab" disabled>UMAP</button>
              </nav>
              <div class="projector-tab-panels">
                <div class="projector-tab-panel is-active" data-tab="pca">
                  <fieldset class="projector-dim">
                    <legend>Dimensions</legend>
                    <label><input type="radio" name="proj-dim" value="2"> 2D</label>
                    <label><input type="radio" name="proj-dim" value="3" checked> 3D</label>
                  </fieldset>
                  <p class="projector-hint">PCA reduces using the top components by variance.</p>
                </div>
              </div>
              <div class="projector-actions">
                <button type="button" class="btn btn-primary" id="proj-run">Run</button>
                <button type="button" class="btn" id="proj-rerun" disabled>Re-run</button>
                <button type="button" class="btn btn-danger" id="proj-cancel" disabled>Stop</button>
              </div>
              <div class="projector-iter" id="proj-iter">iter 0 / 0</div>
            </div>
          </section>
          <section class="projector-panel" id="panel-inspector">
            <header class="projector-panel-title">Inspector</header>
            <div class="projector-panel-body">
              <p class="projector-hint">Search and selection arrive in a later iteration.</p>
            </div>
          </section>
          <section class="projector-panel" id="panel-bookmarks">
            <header class="projector-panel-title">Bookmarks</header>
            <div class="projector-panel-body">
              <p class="projector-hint">Saved views arrive in a later iteration.</p>
            </div>
          </section>
        </aside>
      </section>
    `
)
