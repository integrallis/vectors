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

# Copy the full source tree. The .dockerignore excludes build artifacts, and this keeps the image
# build in sync with settings.gradle.kts as modules are added (rather than a stale curated list).
COPY . ./

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
