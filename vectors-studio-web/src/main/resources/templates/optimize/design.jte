@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param CollectionSummary summary
@template.layout(
    title = summary.name() + " — Optimize",
    head = @`
      <script type="module" src="/static/optimize/design.js" defer></script>
    `,
    body = @`
      <nav class="breadcrumb">
        <a href="/collections">Collections</a>
        <span class="sep">/</span>
        <a href="/collections/${summary.name()}">${summary.name()}</a>
        <span class="sep">/</span>
        <span>Optimize</span>
      </nav>
      <header class="page-header">
        <h1>Optimize <span class="collection-name">${summary.name()}</span></h1>
        <p class="muted">Run a search-time hyperparameter sweep against this collection. Each trial builds a fresh index from the corpus, scores it against a sample of queries, and reports recall, NDCG, and latency.</p>
      </header>

      <div class="optimize-shell" data-collection="${summary.name()}">
        <nav class="study-tabs" role="tablist">
          <button class="tab active" type="button" data-tab="index" role="tab" aria-selected="true">Index</button>
          <button class="tab" type="button" data-tab="router" role="tab" aria-selected="false" disabled title="Coming soon">Router</button>
          <button class="tab" type="button" data-tab="cache" role="tab" aria-selected="false" disabled title="Coming soon">Cache</button>
        </nav>

        <form id="study-form" class="study-form">
          <input type="hidden" name="collection" value="${summary.name()}">

          <section class="card">
            <header class="card-header">
              <h2>Sampler</h2>
              <p class="muted">Pick the search strategy for the parameter space.</p>
            </header>
            <div class="sampler-choices" role="radiogroup">
              <label class="choice">
                <input type="radio" name="sampler" value="GRID">
                <span class="choice-title">Grid</span>
                <span class="choice-help">Exhaustive Cartesian product. Best for small discrete spaces.</span>
              </label>
              <label class="choice selected">
                <input type="radio" name="sampler" value="RANDOM" checked>
                <span class="choice-title">Random</span>
                <span class="choice-help">Uniform per-axis draws. Best baseline; works for any axis type.</span>
              </label>
              <label class="choice">
                <input type="radio" name="sampler" value="TPE">
                <span class="choice-title">TPE</span>
                <span class="choice-help">Tree-structured Parzen Estimator. Best for expensive trials with budget &gt; 20.</span>
              </label>
            </div>
          </section>

          <section class="card">
            <header class="card-header"><h2>Trial budget</h2></header>
            <div class="grid-2">
              <label>Trials <input name="nTrials" type="number" value="12" min="1" max="200"></label>
              <label>k for metrics <input name="kForMetrics" type="number" value="10" min="1" max="100"></label>
              <label>Seed <input name="seed" type="number" value="0" step="1"></label>
              <label>Query sample <input name="querySampleSize" type="number" value="20" min="1" max="500"></label>
            </div>
          </section>

          <section class="card">
            <header class="card-header">
              <h2>Search space</h2>
              <p class="muted">HNSW build-time parameters. Ranges sample log-uniformly so small and large values get equal attention.</p>
            </header>
            <div class="range-row">
              <label class="range-min">m min <input name="mMin" type="number" value="8" min="2" max="256"></label>
              <label class="range-max">m max <input name="mMax" type="number" value="64" min="2" max="256"></label>
            </div>
            <div class="range-row">
              <label class="range-min">efConstruction min <input name="efMin" type="number" value="50" min="10" max="2000"></label>
              <label class="range-max">efConstruction max <input name="efMax" type="number" value="400" min="10" max="2000"></label>
            </div>
          </section>

          <section class="card">
            <header class="card-header">
              <h2>Objective</h2>
              <p class="muted">Weights for the composite score. Cost axes (latency) are subtracted.</p>
            </header>
            <div class="grid-2">
              <label>Recall@k weight <input name="recallWeight" type="number" value="1.0" step="0.1" min="0" max="10"></label>
              <label>NDCG@k weight <input name="ndcgWeight" type="number" value="0.5" step="0.1" min="0" max="10"></label>
              <label>p95 latency weight <input name="latencyWeight" type="number" value="0.0" step="0.1" min="0" max="10"></label>
              <label>Metadata field for qrels <input name="metadataField" type="text" placeholder="optional, e.g. label"></label>
            </div>
          </section>

          <div class="form-actions">
            <button type="submit" class="btn btn-primary">Start study</button>
            <a class="btn btn-link" href="/collections/${summary.name()}">Cancel</a>
          </div>
        </form>
      </div>
    `
)
