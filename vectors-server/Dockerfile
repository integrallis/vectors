# syntax=docker/dockerfile:1.7
#
# Multi-stage build for vectors-server. Build context should be the `vectors/`
# directory (the Gradle multi-project root); invoke from there:
#
#     docker build -t vectors-server:local -f vectors-server/Dockerfile .
#
# The built image runs as non-root UID 1000, listens on 8287, and persists
# collections under /data (declared as a VOLUME). Healthcheck polls /v1/health.

# --------------------------------------------------------------------
# Build stage — full JDK + Gradle wrapper. Skips the expensive verifier
# tasks that already run in CI.
# --------------------------------------------------------------------
FROM eclipse-temurin:25-jdk AS builder

ENV GRADLE_USER_HOME=/gradle-home
WORKDIR /src

# Copy build metadata first so Gradle dependency cache layers when only
# source files change.
COPY settings.gradle.kts build.gradle.kts gradle.properties* ./
COPY gradle ./gradle
COPY gradlew ./gradlew
RUN chmod +x ./gradlew

# Copy the modules needed to compile vectors-server.
COPY vectors-core ./vectors-core
COPY vectors-storage ./vectors-storage
COPY vectors-quantization ./vectors-quantization
COPY vectors-hnsw ./vectors-hnsw
COPY vectors-vamana ./vectors-vamana
COPY vectors-ivf ./vectors-ivf
COPY vectors-db ./vectors-db
COPY vectors-gpu ./vectors-gpu
COPY vectors-distributed ./vectors-distributed
COPY vectors-hybrid ./vectors-hybrid
COPY vectors-router ./vectors-router
COPY vectors-text-h2 ./vectors-text-h2
COPY vectors-spring-ai ./vectors-spring-ai
COPY vectors-spring-boot-starter ./vectors-spring-boot-starter
COPY vectors-langchain4j ./vectors-langchain4j
COPY vectors-bench ./vectors-bench
COPY vectors-server-client ./vectors-server-client
COPY vectors-cache ./vectors-cache
COPY vectors-cache-jcache ./vectors-cache-jcache
COPY vectors-cache-semantic-db ./vectors-cache-semantic-db
COPY vectors-cache-spring-ai ./vectors-cache-spring-ai
COPY vectors-cache-langchain4j ./vectors-cache-langchain4j
COPY vectors-vcr-core ./vectors-vcr-core
COPY vectors-vcr-semantic-db ./vectors-vcr-semantic-db
COPY vectors-vcr-serde-avaje ./vectors-vcr-serde-avaje
COPY vectors-vcr-serde-jackson ./vectors-vcr-serde-jackson
COPY vectors-vcr-junit5 ./vectors-vcr-junit5
COPY vectors-vcr-testng ./vectors-vcr-testng
COPY vectors-vcr-spring-ai ./vectors-vcr-spring-ai
COPY vectors-vcr-langchain4j ./vectors-vcr-langchain4j
COPY vectors-studio-core ./vectors-studio-core
COPY vectors-studio-sidecart ./vectors-studio-sidecart
COPY vectors-studio-web ./vectors-studio-web
COPY vectors-server ./vectors-server
COPY demos ./demos
COPY docs ./docs
COPY spotbugs-exclude.xml ./

RUN ./gradlew :vectors-server:installDist --no-daemon \
        -x test -x spotbugsMain -x spotbugsTest -x spotlessCheck

# --------------------------------------------------------------------
# Runtime stage — slim JDK (required for incubator Vector API) + curl
# for the healthcheck.
# --------------------------------------------------------------------
FROM eclipse-temurin:25-jdk

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --user-group vectors \
    && mkdir -p /data \
    && chown vectors:vectors /data

COPY --from=builder /src/vectors-server/build/install/vectors-server /app
RUN chown -R vectors:vectors /app

USER vectors
WORKDIR /app
EXPOSE 8287
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8287/v1/health || exit 1

ENTRYPOINT ["/app/bin/vectors-server"]
CMD ["--port=8287", "--data-dir=/data"]
