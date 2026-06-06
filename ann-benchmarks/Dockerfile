# ANN-Benchmarks image for `vectors`. Built by the harness as:
#   python install.py --algorithm vectors
# which runs `docker build -f ann_benchmarks/algorithms/vectors/Dockerfile .`
# with the build context at the ann-benchmarks repo root.

# JDK 25 — vectors-server launches with `--add-modules jdk.incubator.vector`
# (the SIMD Vector API), which ships inside the JDK. Copied from the official
# Temurin image (no apt repo setup, network-light, the same image the project's
# own vectors-server Dockerfile uses).
FROM eclipse-temurin:25-jdk AS jdk

FROM ann-benchmarks
COPY --from=jdk /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="/opt/java/openjdk/bin:${PATH}"

# Prebuilt vectors-server distribution. Build it on the host first:
#     ./gradlew :vectors-server:installDist
# then copy vectors-server/build/install/vectors-server into
#     ann_benchmarks/algorithms/vectors/vectors-server-dist/
# (see README.md for the one-liner and the build-from-source alternative).
COPY ann_benchmarks/algorithms/vectors/vectors-server-dist /opt/vectors-server
ENV VECTORS_SERVER_CMD=/opt/vectors-server/bin/vectors-server
