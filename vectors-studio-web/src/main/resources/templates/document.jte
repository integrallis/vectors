@import com.integrallis.vectors.studio.core.search.DocumentView
@param String name
@param DocumentView doc
@template.layout(
    title = "doc " + doc.id(),
    body = @`
      <h1>${doc.id()}</h1>
      <p><a href="/collections/${name}">← back to ${name}</a></p>
      <h2>Text</h2>
      <pre>${doc.text() == null ? "(none)" : doc.text()}</pre>
      <h2>Metadata</h2>
      <pre>${doc.metadata() == null ? "{}" : String.valueOf(doc.metadata())}</pre>
      <h2>Vector</h2>
      <p class="muted">d=${doc.vector() == null ? 0 : doc.vector().length}</p>
    `
)
