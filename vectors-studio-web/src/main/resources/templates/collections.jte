@import java.util.List
@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param List<CollectionSummary> collections
@template.layout(
    title = "Collections",
    body = @`
      <h1>Collections</h1>
      <table class="collection-table">
        <thead><tr><th>Name</th><th>Dim</th><th>Size</th><th>Metric</th><th>Index</th></tr></thead>
        <tbody>
          @for(CollectionSummary c : collections)
            <tr>
              <td><a href="/collections/${c.name()}">${c.name()}</a></td>
              <td>${c.dimension()}</td>
              <td>${c.size()}</td>
              <td>${c.metric()}</td>
              <td>${c.indexType()}</td>
            </tr>
          @endfor
        </tbody>
      </table>
    `
)
