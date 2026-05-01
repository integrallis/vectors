@import com.integrallis.vectors.studio.core.search.ContentKind
@import com.integrallis.vectors.studio.core.search.DocumentView
@import com.integrallis.vectors.studio.web.view.TemplateSupport
@param String name
@param DocumentView doc

!{ContentKind kind = ContentKind.detect(doc);}
!{int dim = doc.vector() == null ? 0 : doc.vector().length;}

@template.layout(
    title = "doc " + doc.id(),
    body = @`
      <nav class="breadcrumb">
        <a href="/collections">Collections</a>
        <span>›</span>
        <a href="/collections/${name}">${name}</a>
        <span>›</span>
        <span class="mono">${doc.id()}</span>
      </nav>

      <header class="doc-header">
        <h1 class="mono">${doc.id()}</h1>
        <span class="kind-badge kind-${kind.name().toLowerCase()}">${kind.name()}</span>
      </header>

      <section class="viewer">
        @if(kind == ContentKind.IMAGE)
          <figure class="image-viewer">
            <img src="/collections/${name}/blobs/${doc.id()}" alt="document ${doc.id()}"
                 loading="lazy"
                 onerror="this.replaceWith(Object.assign(document.createElement('p'),{className:'muted',textContent:'(image not available — no blob registered for this id)'}))">
            @if(doc.text() != null && !doc.text().isEmpty())
              <figcaption class="muted">${TemplateSupport.truncate(doc.text(), 600)}</figcaption>
            @endif
          </figure>
        @elseif(kind == ContentKind.AUDIO)
          <audio controls preload="metadata" src="/collections/${name}/blobs/${doc.id()}"></audio>
        @elseif(kind == ContentKind.BINARY)
          <p class="muted">Binary payload —
            <a class="btn" href="/collections/${name}/blobs/${doc.id()}" download>download</a>
          </p>
        @elseif(kind == ContentKind.JSON)
          <pre class="code-viewer json"><code>${TemplateSupport.prettyJson(doc.text())}</code></pre>
        @elseif(kind == ContentKind.EMPTY)
          <p class="muted">(no text payload)</p>
        @else
          <pre class="code-viewer text"><code>${doc.text()}</code></pre>
        @endif
      </section>

      <section>
        <h2>Metadata</h2>
        <pre class="code-viewer json"><code>${TemplateSupport.prettyJson(doc.metadata())}</code></pre>
      </section>

      <section>
        <h2>Vector</h2>
        <p class="muted">d=${dim}</p>
      </section>
    `
)
