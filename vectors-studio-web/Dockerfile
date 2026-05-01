# syntax=docker/dockerfile:1.7
#
# Multi-stage build for vectors-studio-web. Build context should be the
# `vectors/` directory (the Gradle multi-project root); invoke from there:
#
#     docker build -t vectors-studio-web:local -f vectors-studio-web/Dockerfile .
#
# The built image runs as non-root UID 10001, listens on 8288, and connects to
# a remote vectors-server via --connection http://host:port.
# Healthcheck polls GET /health.

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

# Copy all modules needed to compile vectors-studio-web and its transitive
# dependencies (core → storage → quantization → {hnsw,vamana,ivf} → db →
# studio-core → studio-web, plus server-client for remote backend).
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
COPY vectors-server ./vectors-server
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
COPY demos ./demos
COPY docs ./docs
COPY spotbugs-exclude.xml ./

RUN ./gradlew :vectors-studio-web:installDist --no-daemon \
        -x test -x spotbugsMain -x spotbugsTest -x spotlessCheck

# --------------------------------------------------------------------
# Runtime stage — slim JDK (required for incubator Vector API) + curl
# for the healthcheck.
# --------------------------------------------------------------------
FROM eclipse-temurin:25-jdk

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --user-group vectors

COPY --from=builder /src/vectors-studio-web/build/install/vectors-studio-web /app
RUN chown -R vectors:vectors /app

USER vectors
WORKDIR /app
EXPOSE 8288

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8288/health || exit 1

ENTRYPOINT ["/app/bin/vectors-studio-web"]
CMD ["--port=8288", "--connection=http://vectors-server:8287"]
