# Base image: Python 3.12 + uv preinstalled (Debian slim), pinned by digest
# (of the multi-arch index) so a rebuilt release gets the same base. Up to
# 0.17.4 this was the floating `python3.12-bookworm-slim`, which has pointed
# at this image since uv 0.9.30. To move it, take the index digest of the new
# tag:  docker buildx imagetools inspect ghcr.io/astral-sh/uv:<tag>
FROM ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

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
#
# Checked against the SHA-256 of each release zip, which the build fails
# without. duckdb publishes no checksums for 1.1.3, so these were taken from
# the zips, as served by both GitHub and install.duckdb.org. A new version
# needs both hashes replaced: `sha256sum duckdb_cli-linux-*.zip`.
ARG DUCKDB_VERSION=v1.1.3
ARG DUCKDB_SHA256_AMD64=efd0fccdb1a28d9ec7a6ebfcde59900068b8ba43a846c9b553c0fd2bbe4acf43
ARG DUCKDB_SHA256_AARCH64=6e6caaab69a9c8d77c4400e88197695b072a98b678f6f8ffb015af4f02b13618
RUN ARCH="$(dpkg --print-architecture)" \
    && case "$ARCH" in \
         amd64) DUCKDB_ARCH=amd64; DUCKDB_SHA256="$DUCKDB_SHA256_AMD64" ;; \
         arm64) DUCKDB_ARCH=aarch64; DUCKDB_SHA256="$DUCKDB_SHA256_AARCH64" ;; \
         *) echo "unsupported architecture: $ARCH" >&2; exit 1 ;; \
       esac \
    && curl -fsSL -o /tmp/duckdb.zip \
         "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/duckdb_cli-linux-${DUCKDB_ARCH}.zip" \
    && echo "${DUCKDB_SHA256}  /tmp/duckdb.zip" | sha256sum -c - \
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
#
# The `pickle` (pandas, scipy) and `spatial` (pyarrow) extras are in: a
# pipeline cannot add them to the image later, and without them pycisTopic
# pickles and Visium HD do not convert.
RUN uv sync --locked --extra pickle --extra spatial \
    && rm -rf /tmp/.cache /tmp/uv-*.lock \
    && mkdir -p /tmp/.cache \
    && chmod 1777 /tmp/.cache

# Put the project venv on PATH so `adata` is directly runnable
ENV PATH="/cli/.venv/bin:${PATH}"

# Not root: up to 0.17.4 a plain `docker run` ran as root. Singularity and
# Apptainer run the image as the caller whatever this says, as does Docker
# given `-u $(id -u):$(id -g)`, and everything above is readable by any user
# id. A bind mount the image writes to must be writable by whichever user it
# runs as. Numeric, so that a runtime enforcing non-root (Kubernetes'
# runAsNonRoot) can check it.
RUN useradd --uid 10001 --user-group --create-home --no-log-init adata
USER 10001:10001
WORKDIR /home/adata

# No ENTRYPOINT on purpose: Nextflow requires /bin/bash to be the container
# entrypoint, so the image must not set one of its own. This is why invocations
# spell out the command: `docker run IMAGE adata view file.h5ad`.
#
# Deliberately NOT set here: OMP_NUM_THREADS / OPENBLAS_NUM_THREADS. NumPy's
# BLAS sizes its thread pool to the whole host, which oversubscribes a shared
# LSF node. This workload is streaming I/O, so capping it would cost nothing --
# but it belongs in the pipeline's `env` scope, not baked into the image.
CMD ["adata", "--help"]
