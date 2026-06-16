# Apache Airflow with Java (for PySpark) and project Python dependencies.
FROM apache/airflow:2.9.3

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Java is required by PySpark. procps/bash are convenience tools.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/*

# JAVA_HOME for Spark (architecture-aware: works on amd64 and arm64).
RUN ARCH=$(dpkg --print-architecture) && \
    echo "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-${ARCH}" >> /etc/profile.d/java.sh
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# Fallback symlink so JAVA_HOME resolves on arm64 (e.g. Apple Silicon) too.
RUN if [ ! -d "$JAVA_HOME" ]; then \
        ln -s "$(dirname $(dirname $(readlink -f $(which java))))" "$JAVA_HOME"; \
    fi
ENV PATH="${PATH}:${JAVA_HOME}/bin"

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
