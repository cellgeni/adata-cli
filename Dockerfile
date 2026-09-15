# Base image: Python 3.12 + uv preinstalled (Debian slim)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_NO_DEV=1

WORKDIR /cli

# Copy the project files (from the GitHub Actions checkout context)
COPY . .

# --locked asserts that uv.lock is in sync with pyproject.toml, so an image
# can never be built from a lockfile that drifted.
RUN uv sync --locked

# duckdb, for the filtering workflows in the docs: export obs to CSV, query it,
# feed the names back to `adata subset --obs`. A single static binary, so it
# needs no venv and cannot conflict with the project's dependencies.
ARG DUCKDB_VERSION=v1.1.3
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && ARCH="$(dpkg --print-architecture)" \
    && case "$ARCH" in \
         amd64) DUCKDB_ARCH=amd64 ;; \
         arm64) DUCKDB_ARCH=aarch64 ;; \
         *) echo "unsupported architecture: $ARCH" >&2; exit 1 ;; \
       esac \
    && curl -fsSL -o /tmp/duckdb.zip \
         "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/duckdb_cli-linux-${DUCKDB_ARCH}.zip" \
    && unzip -q /tmp/duckdb.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/duckdb \
    && rm /tmp/duckdb.zip \
    && apt-get purge -y curl unzip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Put the project venv on PATH so `adata` is directly runnable
ENV PATH="/cli/.venv/bin:${PATH}"

ENTRYPOINT ["adata"]
CMD ["--help"]
