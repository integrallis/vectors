@import java.util.List
@import java.util.Set
@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param List<CollectionSummary> collections
@param Set<String> sampleIds
@template.layout(
    title = "Collections",
    body = @`
      <div class="page-head">
        <h1>Collections</h1>
        <a class="btn" href="/datasets">Browse sample datasets</a>
      </div>
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
                <td>
                  <a href="/collections/${c.name()}">${c.name()}</a>
                  @if(sampleIds.contains(c.name()))
                    <span class="pill pill-sample" title="loaded from the sample-dataset catalog">sample</span>
                  @endif
                </td>
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
