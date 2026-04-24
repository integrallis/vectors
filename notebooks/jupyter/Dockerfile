# JDK 25 + Jupyter Lab + DFLib JJava kernel for the java-vectors notebooks.
#
# Base image: jupyter/base-notebook ships Python 3.12 + JupyterLab + a non-root
# 'jovyan' user. We layer Temurin JDK 25 and the JJava 1.0-a7 kernel on top.

FROM quay.io/jupyter/base-notebook:python-3.12

USER root

# --- JDK 25 (Eclipse Temurin) -------------------------------------------------
# Eclipse Adoptium apt repo.
RUN apt-get update && \
    apt-get install --no-install-recommends -y wget gpg ca-certificates unzip && \
    wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | \
        gpg --dearmor -o /usr/share/keyrings/adoptium.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print $2}' /etc/os-release) main" \
        > /etc/apt/sources.list.d/adoptium.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y temurin-25-jdk && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s "$(ls -d /usr/lib/jvm/temurin-25-jdk-* | head -n1)" /usr/lib/jvm/temurin-25-jdk

ENV JAVA_HOME=/usr/lib/jvm/temurin-25-jdk
ENV PATH="$JAVA_HOME/bin:$PATH"

# --- JJava kernel -------------------------------------------------------------
ARG JJAVA_VERSION=1.0-a7
RUN cd /tmp && \
    wget -q "https://github.com/dflib/jjava/releases/download/${JJAVA_VERSION}/jjava-${JJAVA_VERSION}-kernelspec.zip" && \
    unzip -q "jjava-${JJAVA_VERSION}-kernelspec.zip" -d jjava-kernel && \
    jupyter kernelspec install jjava-kernel --name=java --prefix=/opt/conda && \
    rm -rf /tmp/jjava-kernel "/tmp/jjava-${JJAVA_VERSION}-kernelspec.zip"

# Replace the installed kernel.json with our customized version that adds the
# Vector API incubator and native access flags to the JShell JVM.
COPY jupyter/kernel.json /opt/conda/share/jupyter/kernels/java/kernel.json

# `jupyter kernelspec install` (run as root) leaves /home/jovyan/.local and
# /home/jovyan/.ipython owned by root, which breaks JUPYTER_RUNTIME_DIR on
# first launch. Repair the ownership before dropping to NB_UID.
RUN chown -R ${NB_UID}:${NB_GID} /home/jovyan

USER ${NB_UID}

# Workspace is mounted in from docker-compose; working dir is set there.
