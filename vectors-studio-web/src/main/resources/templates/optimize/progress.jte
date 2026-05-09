@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param CollectionSummary summary
@param String studyId
@template.layout(
    title = summary.name() + " — Study " + studyId,
    head = @`
      <script type="module" src="/static/optimize/progress.js" defer></script>
    `,
    body = @`
      <nav class="breadcrumb">
        <a href="/collections">Collections</a>
        <span class="sep">/</span>
        <a href="/collections/${summary.name()}">${summary.name()}</a>
        <span class="sep">/</span>
        <a href="/collections/${summary.name()}/optimize">Optimize</a>
        <span class="sep">/</span>
        <span>Study</span>
      </nav>

      <section id="optimize-progress" class="progress-shell"
               data-study-id="${studyId}"
               data-collection="${summary.name()}">
        <header class="progress-header">
          <div>
            <h1 class="page-title">Study in progress</h1>
            <p class="muted">id: <code>${studyId}</code></p>
          </div>
          <div class="progress-stats">
            <div class="stat"><span class="stat-label">State</span><span class="stat-value" data-bind="state">PENDING</span></div>
            <div class="stat"><span class="stat-label">Trials</span><span class="stat-value"><span data-bind="trials-completed">0</span> / <span data-bind="trials-total">0</span></span></div>
            <div class="stat"><span class="stat-label">Best score</span><span class="stat-value" data-bind="best-score">—</span></div>
            <div class="stat"><span class="stat-label">Elapsed</span><span class="stat-value" data-bind="elapsed">0s</span></div>
          </div>
          <div class="progress-actions">
            <button type="button" class="btn btn-danger" data-action="cancel">Cancel</button>
          </div>
        </header>

        <section class="card">
          <header class="card-header">
            <h2>Optimization history</h2>
            <p class="muted">Composite objective per trial; running best in solid line.</p>
          </header>
          <canvas id="history-canvas" width="900" height="280" aria-label="Optimization history line chart"></canvas>
        </section>

        <section class="card">
          <header class="card-header">
            <h2>Trials</h2>
            <p class="muted">Click a column header to sort. Click <em>Apply</em> on a trial to capture its parameters.</p>
          </header>
          <div class="trial-table-wrap">
            <table class="trial-table" id="trial-table">
              <thead>
                <tr>
                  <th data-sort="index">#</th>
                  <th data-sort="trialId">Trial</th>
                  <th data-sort="m">m</th>
                  <th data-sort="ef">efConstruction</th>
                  <th data-sort="recall">Recall@k</th>
                  <th data-sort="ndcg">NDCG@k</th>
                  <th data-sort="latency">p95 µs</th>
                  <th data-sort="build">Build ms</th>
                  <th data-sort="objective">Score</th>
                  <th></th>
                </tr>
              </thead>
              <tbody id="trial-rows"></tbody>
            </table>
          </div>
        </section>
      </section>
    `
)
