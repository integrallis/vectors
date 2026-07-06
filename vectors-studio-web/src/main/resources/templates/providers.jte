@import java.util.List
@import com.integrallis.vectors.studio.web.embed.ProviderStatus
@param List<ProviderStatus> providers
@param String overrideEnv
@template.layout(
    title = "Providers",
    body = @`
      <div class="page-head">
        <h1>Providers</h1>
        <a class="btn" href="/collections">Back to collections</a>
      </div>
      <p class="muted">
        Embedding providers turn free-text search queries into vectors. Each collection loaded from
        the sample catalog records the model its vectors were computed with; to search it by text you
        need a provider here that serves the matching query model with its key present.
      </p>
      <table class="collection-table">
        <thead>
          <tr>
            <th>Id</th><th>Type</th><th>Models</th><th>Key status</th>
          </tr>
        </thead>
        <tbody>
          @for(ProviderStatus p : providers)
            <tr>
              <td>${p.id()}</td>
              <td><span class="pill pill-metric">${p.type()}</span></td>
              <td class="muted">${String.join(", ", p.models())}</td>
              <td>
                @if(p.keyPresent())
                  <span class="pill pill-sample" title="key present or none required">✓ ready</span>
                @else
                  <span class="form-error" title="set the environment variable to enable this provider">✗ missing key</span>
                @endif
                @if(p.apiKeyEnv() != null)
                  <div class="muted">env: <code>${p.apiKeyEnv()}</code></div>
                @else
                  <div class="muted">no key required</div>
                @endif
              </td>
            </tr>
          @endfor
        </tbody>
      </table>
      <p class="muted">
        Keys are read from the process environment, JVM system properties, or a gitignored
        <code>.env</code> at the repo root — only the presence of a key is shown here, never its
        value. To add or override a provider, point <code>${overrideEnv}</code> at a JSON file with
        the same schema as the bundled <code>embedding-providers.json</code>; entries are merged by
        <code>id</code>.
      </p>
    `
)
