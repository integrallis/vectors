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

# Copy the full source tree. The .dockerignore excludes build artifacts, and this keeps the image
# build in sync with settings.gradle.kts as modules are added (rather than a stale curated list).
COPY . ./

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
