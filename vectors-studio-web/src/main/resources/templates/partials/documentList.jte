@import java.util.List
@import com.integrallis.vectors.studio.core.search.DocumentView
@param String name
@param List<DocumentView> page
@param int offset
@param int limit
<table class="doc-list">
  <thead><tr><th>id</th><th>text</th></tr></thead>
  <tbody>
    @for(DocumentView d : page)
      <tr>
        <td><a href="/collections/${name}/documents/${d.id()}">${d.id()}</a></td>
        <td>${d.text() == null ? "" : d.text()}</td>
      </tr>
    @endfor
  </tbody>
</table>
<nav class="pager">
  @if(offset > 0)
    <button hx-get="/collections/${name}/preview?offset=${Math.max(0, offset - limit)}&limit=${limit}" hx-target="#preview-pane">Prev</button>
  @endif
  @if(page.size() == limit)
    <button hx-get="/collections/${name}/preview?offset=${offset + limit}&limit=${limit}" hx-target="#preview-pane">Next</button>
  @endif
</nav>
