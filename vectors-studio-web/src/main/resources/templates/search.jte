@import java.util.List
@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@import com.integrallis.vectors.studio.core.search.SearchHit
@param CollectionSummary summary
@param List<SearchHit> hits
@param String query = ""
@template.layout(
    title = "Search " + summary.name(),
    body = @`
      <h1>Search ${summary.name()}</h1>
      <form class="search-form" hx-post="/collections/${summary.name()}/search" hx-target="#hits-pane">
        <label class="field-query">Query <input type="text" name="query" value="${query == null ? "" : query}" required></label>
        <label class="field-k">k <input name="k" type="number" value="10" min="1" max="100"></label>
        <button type="submit">Search</button>
      </form>
      <div id="hits-pane">
        @template.partials.hitsList(hits = hits)
      </div>
    `
)
