@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param CollectionSummary summary
@template.layout(
    title = summary.name(),
    body = @`
      <h1>${summary.name()}</h1>
      <p class="muted">d=${summary.dimension()} · ${summary.size()} docs · metric=${summary.metric()} · index=${summary.indexType()}</p>
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
