@import java.util.List
@import com.integrallis.vectors.studio.core.search.SearchHit
@param List<SearchHit> hits
@if(hits == null || hits.isEmpty())
  <p class="empty-state">No results.</p>
@else
  <table class="hits">
    <thead><tr><th>id</th><th class="col-score">score</th><th>text</th></tr></thead>
    <tbody>
      @for(SearchHit h : hits)
        <tr>
          <td class="mono">${h.id()}</td>
          <td class="col-score mono">${String.format("%.4f", h.score())}</td>
          <td>${h.text() == null ? "" : h.text()}</td>
        </tr>
      @endfor
    </tbody>
  </table>
@endif
