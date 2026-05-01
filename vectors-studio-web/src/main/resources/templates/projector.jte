@import com.integrallis.vectors.studio.core.connection.CollectionSummary
@param CollectionSummary summary
@template.layout(
    title = "Projector — " + summary.name(),
    body = @`
      <style>
        /* Break out of main's padding and max-width for full-screen projector */
        .projector-wrap {
          position: fixed;
          top: 56px; /* below the app header */
          left: 0;
          right: 0;
          bottom: 0;
        }
        .projector-wrap iframe {
          width: 100%;
          height: 100%;
          border: none;
        }
      </style>
      <div class="projector-wrap">
        <iframe src="/static/projector/index.html?config=/api/collections/${summary.name()}/projector-config"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope"
                allowfullscreen>
        </iframe>
      </div>
    `
)
