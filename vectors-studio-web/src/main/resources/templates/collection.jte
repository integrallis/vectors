@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param CollectionSummary summary
@template.layout(
    title = summary.name(),
    body = @`
      <h1>${summary.name()}</h1>
      <div class="collection-meta">
        <span class="pill pill-dim" title="dimension">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M5 3v16h16"/>
            <path d="m5 19 6-6"/>
            <path d="m2 6 3-3 3 3"/>
            <path d="m18 16 3 3-3 3"/>
          </svg>
          d=${summary.dimension()}
        </span>
        <span class="pill pill-docs" title="documents">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M20 7h-3a2 2 0 0 1-2-2V2"/>
            <path d="M9 18a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h7l4 4v10a2 2 0 0 1-2 2Z"/>
            <path d="M3 7.6v12.8A1.6 1.6 0 0 0 4.6 22h9.8"/>
          </svg>
          ${summary.size()} docs
        </span>
        <span class="pill pill-metric" title="metric">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="10"/>
            <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
          </svg>
          ${summary.metric()}
        </span>
        <span class="pill pill-index" title="index">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polygon points="12 2 2 7 12 12 22 7 12 2"/>
            <polyline points="2 17 12 22 22 17"/>
            <polyline points="2 12 12 17 22 12"/>
          </svg>
          ${summary.indexType()}
        </span>
      </div>
      <nav class="actions">
        <a class="btn btn-icon" href="/collections/${summary.name()}/projector" title="Visualize as a 3D point cloud">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
            <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
            <line x1="12" y1="22.08" x2="12" y2="12"/>
          </svg>
          <span>Visualize</span>
        </a>
        <a class="btn btn-icon" href="/collections/${summary.name()}/optimize" title="Tune index, router, and cache parameters">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <line x1="4" y1="21" x2="4" y2="14"/>
            <line x1="4" y1="10" x2="4" y2="3"/>
            <line x1="12" y1="21" x2="12" y2="12"/>
            <line x1="12" y1="8" x2="12" y2="3"/>
            <line x1="20" y1="21" x2="20" y2="16"/>
            <line x1="20" y1="12" x2="20" y2="3"/>
            <line x1="1" y1="14" x2="7" y2="14"/>
            <line x1="9" y1="8" x2="15" y2="8"/>
            <line x1="17" y1="16" x2="23" y2="16"/>
          </svg>
          <span>Optimize</span>
        </a>
      </nav>
      <h2>Search</h2>
      <form class="search-form" hx-post="/collections/${summary.name()}/search" hx-target="#hits-pane">
        <label class="field-query">Query <input type="text" name="query" placeholder="search…" required></label>
        <label class="field-k">k <input name="k" type="number" value="10" min="1" max="100"></label>
        <button type="submit">Search</button>
      </form>
      <div id="hits-pane"></div>
      <h2>Documents</h2>
      <div id="preview-pane" hx-get="/collections/${summary.name()}/preview?offset=0&limit=25" hx-trigger="load">
        <p class="muted">loading…</p>
      </div>
    `
)
