@import com.integrallis.vectors.studio.core.recommender.DatasetStats
@import com.integrallis.vectors.studio.core.recommender.ProjectionRecommendation
@param String name
@param DatasetStats stats
@param ProjectionRecommendation recommendation
@param boolean llmAvailable
@template.layout(
    title = "Recommend — " + name,
    body = @`
      <h1>Recommended projection — ${name}</h1>
      <h2>${recommendation.algorithm()}</h2>
      <p><strong>Why:</strong> ${recommendation.rationale()}</p>
      <p><strong>Params:</strong> ${String.valueOf(recommendation.params())}</p>
      <h3>Dataset</h3>
      <ul>
        <li>N = ${stats.n()}</li>
        <li>dim = ${stats.dim()}</li>
        <li>intrinsicDim ≈ ${stats.intrinsicDimEstimate()}</li>
        <li>sparsity = ${String.format("%.2f", stats.sparsityRatio())}</li>
      </ul>
      @if(llmAvailable)
        <button hx-get="/collections/${name}/recommend/explain" hx-target="#llm-pane">Explain with AI</button>
        <div id="llm-pane"></div>
      @else
        <p class="muted">LLM enrichment is not configured.</p>
      @endif
    `
)
