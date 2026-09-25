# Studio browser regression check

This uses real Chromium, HTMX, Three.js/WebGL, SSE, Smile projections and the current built
Vectors libraries. The 96 synthetic vectors are a deterministic UI fixture, not a quality or
performance dataset. No provider API keys or remote model calls are required.

Install Python Playwright and its Chromium browser on the test host. Install OpenBLAS and ARPACK as described
in the Studio guide. Run as an ordinary user, then:

```sh
./gradlew :vectors-studio-web:installDist :vectors-studio-web:integrationTest
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  --class-path 'vectors-studio-web/build/install/vectors-studio-web/lib/*' \
  vectors-studio-web/smoke/StudioFixture.java
# In a second shell:
python vectors-studio-web/smoke/browser.py --output build/studio-browser
```

Keep port 8288 private. The fixture is disposable, including its delete-collection check. The test
captures screenshots, a Playwright trace, JavaScript errors, failed requests and a JSON outcome.
External CDN dependencies are exercised as shipped; they are not intercepted or replaced.

This covers collection listing/pagination/empty state/deletion, literal document text and metadata,
PCA/t-SNE/UMAP, 2D, nearest-neighbor inspection/MMR, and the dataset/provider pages. It does not yet
qualify distributed/R2 operation, paid embedding providers, real-dataset loading, or the separate
multimodal RAG UI. Those need their own recorded execution.
