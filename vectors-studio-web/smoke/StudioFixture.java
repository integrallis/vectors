import com.integrallis.vectors.core.Document;
import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.VectorCollection;
import com.integrallis.vectors.studio.core.StudioSession;
import com.integrallis.vectors.studio.core.connection.EmbeddedStudioBackend;
import com.integrallis.vectors.studio.web.StudioConfig;
import com.integrallis.vectors.studio.web.StudioServer;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CountDownLatch;

/** Synthetic UI fixture only: no model-quality or projection-quality claim. */
class StudioFixture {
  public static void main(String[] args) throws Exception {
    var docs = VectorCollection.builder().dimension(8)
        .metric(SimilarityFunction.COSINE).indexType(IndexType.FLAT).build();
    var empty = VectorCollection.builder().dimension(8)
        .metric(SimilarityFunction.COSINE).indexType(IndexType.FLAT).build();
    var random = new Random(20260925);
    for (int i = 0; i < 96; i++) {
      float[] vector = new float[8];
      for (int j = 0; j < vector.length; j++) vector[j] = (float) random.nextGaussian();
      docs.add(new Document("doc-" + i, vector,
          i == 0 ? "Literal <script>window.fixtureInjection=true</script> & text" : "Fixture document " + i,
          Map.of("category", MetadataValue.of("group-" + i % 3), "row", MetadataValue.of((double) i))));
    }
    docs.commit();
    var backend = EmbeddedStudioBackend.withCollections(Map.of("docs", docs, "empty", empty));
    try (var server = StudioServer.start(new StudioConfig(8288, new StudioSession(backend)))) {
      System.out.println("STUDIO_FIXTURE_READY port=" + server.port());
      new CountDownLatch(1).await();
    }
  }
}
