@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param CollectionSummary summary
@template.layout(
    title = "Projector — " + summary.name(),
    body = @`
      <h1>Projector — ${summary.name()}</h1>
      <p class="muted">d=${summary.dimension()} · ${summary.size()} docs</p>
      <form id="proj-form">
        <input type="hidden" name="collection" value="${summary.name()}">
        <label>Algorithm
          <select name="algorithm">
            <option value="PCA">PCA</option>
            <option value="TSNE">t-SNE</option>
            <option value="UMAP">UMAP</option>
          </select>
        </label>
        <label>Dimensions
          <select name="dimensions">
            <option value="2">2D</option>
            <option value="3">3D</option>
          </select>
        </label>
        <label>Sample size <input name="sampleSize" type="number" value="2000" min="100" max="50000"></label>
        <button type="submit">Project</button>
      </form>
      <div id="projector-status" class="muted">idle</div>
      <div id="projector-canvas" style="width:100%;height:500px;border:1px solid #ccc;"></div>
      <script type="module" src="/static/projector.js"></script>
    `
)
