@import java.util.List
@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param List<CollectionSummary> collections
@template.layout(
    title = "Collections",
    body = @`
      <h1>Collections</h1>
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
                <td>${c.dimension()}</td>
                <td>${c.size()}</td>
                <td>${c.metric()}</td>
                <td>${c.indexType()}</td>
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
