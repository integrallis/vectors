@import java.util.List
@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param List<CollectionSummary> collections
@template.layout(
    title = "Collections",
    body = @`
      <h1>Collections</h1>
      <section class="sample-datasets">
        <h2>Sample datasets</h2>
        <div class="dataset-gallery">
          <div class="dataset-card">
            <h3>dbpedia</h3>
            <p class="muted">OpenAI embeddings (Wikipedia)</p>
            <button class="btn" id="load-dbpedia" onclick="loadSampleDataset(this)">Load sample</button>
            <span class="dataset-status muted" id="load-dbpedia-status"></span>
          </div>
        </div>
      </section>
      <script>
        function loadSampleDataset(btn) {
          var status = document.getElementById('load-dbpedia-status');
          btn.disabled = true;
          status.textContent = 'loading…';
          fetch('/api/datasets/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              dataset: 'KShivendu/dbpedia-entities-openai-1M',
              name: 'dbpedia-sample',
              vectorColumn: 'openai',
              textColumn: 'text',
              idColumn: '_id',
              limit: 2000
            })
          }).then(function (r) {
            if (!r.ok) { return r.text().then(function (t) { throw new Error(t || ('HTTP ' + r.status)); }); }
            return r.json();
          }).then(function (data) {
            status.textContent = 'loaded ' + data.size + ' docs';
            window.location.href = '/collections/' + data.name + '/projector';
          }).catch(function (err) {
            status.textContent = 'error: ' + err.message;
            btn.disabled = false;
          });
        }
      </script>
      @if(collections.isEmpty())
        <p class="muted">No collections yet.</p>
      @else
        <table class="collection-table">
          <thead>
            <tr>
              <th>Name</th><th>Dim</th><th>Size</th><th>Metric</th><th>Index</th><th class="actions-col">Actions</th>
            </tr>
          </thead>
          <tbody>
            @for(CollectionSummary c : collections)
              <tr id="row-${c.name()}">
                <td><a href="/collections/${c.name()}">${c.name()}</a></td>
                <td>
                  <span class="pill pill-dim" title="dimension">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                      <path d="M5 3v16h16"/>
                      <path d="m5 19 6-6"/>
                      <path d="m2 6 3-3 3 3"/>
                      <path d="m18 16 3 3-3 3"/>
                    </svg>
                    d=${c.dimension()}
                  </span>
                </td>
                <td>
                  <span class="pill pill-docs" title="documents">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                      <path d="M20 7h-3a2 2 0 0 1-2-2V2"/>
                      <path d="M9 18a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h7l4 4v10a2 2 0 0 1-2 2Z"/>
                      <path d="M3 7.6v12.8A1.6 1.6 0 0 0 4.6 22h9.8"/>
                    </svg>
                    ${c.size()} docs
                  </span>
                </td>
                <td>
                  <span class="pill pill-metric" title="metric">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                      <circle cx="12" cy="12" r="10"/>
                      <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
                    </svg>
                    ${c.metric()}
                  </span>
                </td>
                <td>
                  <span class="pill pill-index" title="index">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                      <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                      <polyline points="2 17 12 22 22 17"/>
                      <polyline points="2 12 12 17 22 12"/>
                    </svg>
                    ${c.indexType()}
                  </span>
                </td>
                <td class="actions-col">
                  <button class="btn btn-danger"
                          hx-delete="/collections/${c.name()}"
                          hx-confirm="Delete collection '${c.name()}'? This cannot be undone."
                          hx-target="#row-${c.name()}"
                          hx-swap="outerHTML">
                    Delete
                  </button>
                </td>
              </tr>
            @endfor
          </tbody>
        </table>
      @endif
    `
)
