@import java.util.List
@import com.integrallis.vectors.studio.core.search.SearchHit
@param List<SearchHit> hits
<table class="hits">
  <thead><tr><th>id</th><th>score</th><th>text</th></tr></thead>
  <tbody>
    @for(SearchHit h : hits)
      <tr>
        <td>${h.id()}</td>
        <td>${String.format("%.4f", h.score())}</td>
        <td>${h.text() == null ? "" : h.text()}</td>
      </tr>
    @endfor
  </tbody>
</table>
