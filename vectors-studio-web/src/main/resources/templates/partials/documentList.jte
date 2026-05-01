@import java.util.List
@import com.integrallis.vectors.studio.core.search.ContentKind
@import com.integrallis.vectors.studio.core.search.DocumentView
@import com.integrallis.vectors.studio.web.view.TemplateSupport
@param String name
@param List<DocumentView> page
@param int offset
@param int limit
@param long total

!{int pageCount = TemplateSupport.pageCount(total, limit);}
!{int currentPage = TemplateSupport.currentPage(offset, limit);}
!{long pageStart = total == 0 ? 0 : (long) offset + 1;}
!{long pageEnd = Math.min((long) offset + page.size(), total);}

<div class="doc-list-wrap">
  <div class="doc-list-header">
    <span class="muted">Showing ${pageStart}–${pageEnd} of ${total}</span>
    <span class="page-size">
      Rows per page:
      @for(int n : new int[]{10, 25, 50, 100})
        <button class="page-size-btn ${n == limit ? "active" : ""}"
                hx-get="/collections/${name}/preview?offset=0&limit=${n}"
                hx-target="#preview-pane">${n}</button>
      @endfor
    </span>
  </div>

  <table class="doc-list">
    <thead>
      <tr>
        <th class="col-id">id</th>
        <th class="col-kind">kind</th>
        <th class="col-preview">preview</th>
      </tr>
    </thead>
    <tbody>
      @if(page.isEmpty())
        <tr><td colspan="3" class="muted">No documents on this page.</td></tr>
      @else
        @for(DocumentView d : page)
          !{ContentKind kind = ContentKind.detect(d);}
          <tr>
            <td class="col-id"><a href="/collections/${name}/documents/${d.id()}" class="mono">${d.id()}</a></td>
            <td class="col-kind"><span class="kind-badge kind-${kind.name().toLowerCase()}">${kind.name()}</span></td>
            <td class="col-preview">
              @if(kind == ContentKind.IMAGE)
                <span class="muted">[image] click id to view</span>
              @elseif(kind == ContentKind.AUDIO)
                <span class="muted">[audio] click id to play</span>
              @elseif(kind == ContentKind.BINARY)
                <span class="muted">[binary] click id to download</span>
              @elseif(kind == ContentKind.EMPTY)
                <span class="muted">(no text)</span>
              @else
                <span class="preview-text">${TemplateSupport.truncate(d.text(), 240)}</span>
              @endif
            </td>
          </tr>
        @endfor
      @endif
    </tbody>
  </table>

  <nav class="pager" aria-label="pagination">
    <button class="pager-btn"
            disabled="${offset == 0}"
            hx-get="/collections/${name}/preview?offset=0&limit=${limit}"
            hx-target="#preview-pane">« First</button>
    <button class="pager-btn"
            disabled="${offset == 0}"
            hx-get="/collections/${name}/preview?offset=${Math.max(0, offset - limit)}&limit=${limit}"
            hx-target="#preview-pane">‹ Prev</button>
    <span class="pager-info">Page ${currentPage + 1} of ${Math.max(1, pageCount)}</span>
    <button class="pager-btn"
            disabled="${currentPage + 1 >= pageCount}"
            hx-get="/collections/${name}/preview?offset=${offset + limit}&limit=${limit}"
            hx-target="#preview-pane">Next ›</button>
    <button class="pager-btn"
            disabled="${currentPage + 1 >= pageCount}"
            hx-get="/collections/${name}/preview?offset=${(pageCount - 1) * limit}&limit=${limit}"
            hx-target="#preview-pane">Last »</button>
  </nav>
</div>
