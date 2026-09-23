# Base image: Python 3.12 + uv preinstalled (Debian slim)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# PYTHONNOUSERSITE: Apptainer bind-mounts the host $HOME by default, so a user's
#   ~/.local/lib/python3.12/site-packages would otherwise shadow this venv.
# XDG_CACHE_HOME: Nextflow is commonly configured with `-u $(id -u):$(id -g)`,
#   which leaves the container with no writable $HOME.
# UV_COMPILE_BYTECODE: bake .pyc at build time, so nothing writes to a
#   read-only rootfs on first import.
ENV UV_NO_DEV=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/tmp/.cache

# procps supplies `ps`, which Nextflow needs to collect per-task metrics.
# curl, unzip and ca-certificates fetch duckdb below, and are left in place
# rather than purged: pipeline scripts routinely reach for curl, and TLS
# roots are worth having in any container that may touch the network.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         ca-certificates curl procps mawk unzip \
    && rm -rf /var/lib/apt/lists/*

# duckdb, for the filtering workflows in the docs: export obs to CSV, query it,
# feed the names back to `adata subset --obs`. A single static binary, so it
# needs no venv and cannot conflict with the project's dependencies.
ARG DUCKDB_VERSION=v1.1.3
RUN ARCH="$(dpkg --print-architecture)" \
    && case "$ARCH" in \
         amd64) DUCKDB_ARCH=amd64 ;; \
         arm64) DUCKDB_ARCH=aarch64 ;; \
         *) echo "unsupported architecture: $ARCH" >&2; exit 1 ;; \
       esac \
    && curl -fsSL -o /tmp/duckdb.zip \
         "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/duckdb_cli-linux-${DUCKDB_ARCH}.zip" \
    && unzip -q /tmp/duckdb.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/duckdb \
    && rm /tmp/duckdb.zip

# Fail the build, rather than every Nextflow task, if the base image ever drops
# one of the tools Nextflow requires in a task container.
RUN set -eu; for t in bash ps awk date grep sed tail tee; do \
      command -v "$t" >/dev/null || { echo "missing required tool: $t" >&2; exit 1; }; \
    done

WORKDIR /cli

# Copy the project files (from the GitHub Actions checkout context)
COPY . .

# --locked asserts that uv.lock is in sync with pyproject.toml, so an image
# can never be built from a lockfile that drifted.
#
# uv honours XDG_CACHE_HOME, so the sync leaves a root-owned package cache at
# /tmp/.cache -- which made the variable self-defeating, as a task running
# under an arbitrary UID then could not write to the very path it advertises.
# Clear it and leave an empty world-writable directory behind. Done in this
# same layer because a later `rm` would mask the files without reclaiming
# them; that reclaims about 9 MB, the cache being mostly hardlinks into the
# venv rather than separate copies.
RUN uv sync --locked \
    && rm -rf /tmp/.cache /tmp/uv-*.lock \
    && mkdir -p /tmp/.cache \
    && chmod 1777 /tmp/.cache

# Put the project venv on PATH so `adata` is directly runnable
ENV PATH="/cli/.venv/bin:${PATH}"

# No ENTRYPOINT on purpose: Nextflow requires /bin/bash to be the container
# entrypoint, so the image must not set one of its own. This is why invocations
# spell out the command: `docker run IMAGE adata view file.h5ad`.
#
# Deliberately NOT set here: OMP_NUM_THREADS / OPENBLAS_NUM_THREADS. NumPy's
# BLAS sizes its thread pool to the whole host, which oversubscribes a shared
# LSF node. This workload is streaming I/O, so capping it would cost nothing --
# but it belongs in the pipeline's `env` scope, not baked into the image.
CMD ["adata", "--help"]
