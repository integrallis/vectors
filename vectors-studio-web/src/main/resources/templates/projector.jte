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
                <button class="projector-tab" data-tab="tsne" role="tab">t-SNE</button>
                <button class="projector-tab" data-tab="umap" role="tab">UMAP</button>
              </nav>
              <div class="projector-tab-panels">
                <div class="projector-tab-panel is-active" data-tab="pca">
                  <p class="projector-hint">PCA reduces using the top components by variance.</p>
                </div>
                <div class="projector-tab-panel" data-tab="tsne">
                  <label class="projector-field projector-field--inline">
                    <span>Perplexity</span>
                    <input type="number" id="tsne-perplexity" min="2" max="100" step="1" value="15">
                  </label>
                  <label class="projector-field projector-field--inline">
                    <span>Learning rate</span>
                    <input type="number" id="tsne-learning-rate" min="1" step="1" value="200">
                  </label>
                  <label class="projector-field projector-field--inline">
                    <span>Max iterations</span>
                    <input type="number" id="tsne-iterations" min="50" step="50" value="1000">
                  </label>
                  <label class="projector-field projector-field--inline">
                    <span>Random seed</span>
                    <input type="number" id="tsne-seed" step="1" value="42">
                  </label>
                  <p class="projector-hint">t-SNE preserves local neighborhoods; results vary by seed.</p>
                </div>
                <div class="projector-tab-panel" data-tab="umap">
                  <label class="projector-field projector-field--inline">
                    <span>Neighbors</span>
                    <input type="number" id="umap-neighbors" min="2" max="200" step="1" value="15">
                  </label>
                  <label class="projector-field projector-field--inline">
                    <span>Min distance</span>
                    <input type="number" id="umap-min-dist" min="0" max="1" step="0.05" value="0.1">
                  </label>
                  <label class="projector-field projector-field--inline">
                    <span>Epochs</span>
                    <input type="number" id="umap-iterations" min="50" step="50" value="200">
                  </label>
                  <label class="projector-field projector-field--inline">
                    <span>Random seed</span>
                    <input type="number" id="umap-seed" step="1" value="42">
                  </label>
                  <p class="projector-hint">UMAP balances local and global structure.</p>
                </div>
              </div>
              <fieldset class="projector-dim">
                <legend>Dimensions</legend>
                <label><input type="radio" name="proj-dim" value="2"> 2D</label>
                <label><input type="radio" name="proj-dim" value="3" checked> 3D</label>
              </fieldset>
              <div class="projector-actions">
                <button type="button" class="btn btn-primary" id="proj-run">Run</button>
                <button type="button" class="btn" id="proj-pause" disabled>Pause</button>
                <button type="button" class="btn" id="proj-rerun" disabled>Re-run</button>
              </div>
              <div class="projector-iter" id="proj-iter">iter 0 / 0</div>
            </div>
          </section>
          <section class="projector-panel" id="panel-inspector">
            <header class="projector-panel-title">Inspector</header>
            <div class="projector-panel-body">
              <label class="projector-field">
                <span>Search by id</span>
                <input type="text" id="ins-query" placeholder="doc-0" autocomplete="off">
              </label>
              <label class="projector-field projector-field--inline">
                <span>Neighbors</span>
                <input type="number" id="ins-k" min="1" max="200" step="1" value="10">
              </label>
              <div class="projector-actions">
                <button type="button" class="btn btn-primary" id="ins-search">Search</button>
                <button type="button" class="btn" id="ins-lasso" aria-pressed="false">Lasso</button>
              </div>
              <ul class="projector-hits" id="ins-hits"></ul>
              <div class="projector-actions">
                <button type="button" class="btn" id="ins-isolate" disabled>Isolate</button>
                <button type="button" class="btn" id="ins-clear" disabled>Clear</button>
                <button type="button" class="btn" id="ins-show-all" disabled>Show All</button>
              </div>
              <div class="projector-iter" id="ins-status"></div>
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
