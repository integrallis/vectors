@import java.util.List
@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param List<CollectionSummary> collections
@template.layout(
    title = "Home",
    body = @`
      <h1>Collections</h1>
      @if(collections.isEmpty())
        <p>No collections found. Create one via the API or point Studio at a populated data dir.</p>
      @else
        <ul class="collection-list">
          @for(CollectionSummary c : collections)
            <li>
              <a href="/collections/${c.name()}">${c.name()}</a>
              <span class="muted">d=${c.dimension()} · ${c.size()} docs · ${c.metric()}</span>
            </li>
          @endfor
        </ul>
      @endif
    `
)
