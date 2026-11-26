#!/usr/bin/env bash
# Check that an adata-cli image runs, before anything pushes it.
#
#     bash .github/scripts/smoke-image.sh IMAGE VERSION
#
# Its help and version, that it runs unprivileged, duckdb, and convert and
# view as the caller on a read-only root filesystem, as Singularity and
# Nextflow run it. tests.yml runs this on every pull request and before a
# release reaches PyPI; quay-on-tag.yml runs it again on the image it pushes.
set -euo pipefail

image=$1
version=$2

docker run --rm "$image" adata --help > /dev/null
reported=$(docker run --rm "$image" adata --version)
if [ "$reported" != "$version" ]; then
  echo "::error::the image reports $reported, the release is $version"
  exit 1
fi
if [ "$(docker run --rm "$image" id -u)" = 0 ]; then
  echo "::error::the image runs as root"
  exit 1
fi
docker run --rm "$image" duckdb -c "select 42"

# Outside the build context, so that a later build of it is not changed.
data=$(mktemp -d "${RUNNER_TEMP:-/tmp}/smoke.XXXXXX")
printf ',g1,g2,g3\nc1,0,1,2\nc2,3,4,5\nc3,6,7,8\n' > "$data/counts.csv"
as_caller() {
  docker run --rm --user "$(id -u):$(id -g)" --read-only --tmpfs /tmp \
    -v "$data":/data "$image" "$@"
}
as_caller adata convert /data/counts.csv -o /data/counts.h5ad
as_caller adata convert /data/counts.h5ad -o /data/counts.zarr
as_caller adata view /data/counts.zarr
