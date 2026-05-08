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
        <a class="btn" href="/collections/${summary.name()}/search">Search</a>
        <a class="btn" href="/collections/${summary.name()}/projector">Projector</a>
        <a class="btn" href="/collections/${summary.name()}/recommend">Recommend projection</a>
      </nav>
      <h2>Documents</h2>
      <div id="preview-pane" hx-get="/collections/${summary.name()}/preview?offset=0&limit=25" hx-trigger="load">
        <p class="muted">loading…</p>
      </div>
    `
)
