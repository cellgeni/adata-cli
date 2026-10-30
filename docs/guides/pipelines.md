---
title: Pipelines and containers
parent: Guides
nav_order: 6
---

# Pipelines and containers
{: .no_toc }

1. TOC
{:toc}

## The container image

Every release is published as `quay.io/cellgeni/adata-cli:<version>`, and the
latest as `:latest`. The image has the `pickle` and `spatial` extras and
[duckdb](https://duckdb.org) in it, and sets no entrypoint, as Nextflow
needs, so name the command:

```bash
docker run --rm -v "$PWD":/data quay.io/cellgeni/adata-cli:0.16.1 \
    adata view /data/data.h5ad

singularity exec docker://quay.io/cellgeni/adata-cli:0.16.1 \
    adata view data.h5ad
```

{: .important }
Pin the version in a pipeline. `:latest` moves with every release, and a
rerun months later would use different code.

The image runs under any user id and on a read-only root filesystem: a
Singularity or Apptainer bind of your home directory cannot shadow its
packages, and Python's caches go to `/tmp`.

In a Nextflow process:

```nextflow
process SUBSET {
    container 'quay.io/cellgeni/adata-cli:0.16.1'
    memory '2 GB'

    input:
    tuple val(id), path(h5ad), path(barcodes)

    output:
    tuple val(id), path("${id}.subset.h5ad")

    script:
    """
    adata subset ${h5ad} -o ${id}.subset.h5ad --obs ${barcodes}
    """
}
```

The image leaves `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` unset. `adata`
is I/O-bound, so setting them to 1 in the pipeline's `env` scope costs nothing
on a shared node and stops numpy sizing its thread pool to the whole machine.

## Outputs appear whole or not at all

Every command that writes a store writes it beside its destination, as
`out.adata-tmp.h5ad`, and renames it into place only when it is complete. An
existing output is first renamed aside to `out.adata-old.h5ad`, and deleted
once the new one is in place. So:

- a failed command leaves no partial output for the next step to read as if
  it were whole, even when the failure is the last flush to disk;
- an existing output, or with `--inplace` the source, survives a failure;
- the temporary store sits in the output's directory, on the same filesystem,
  which needs room for it, not the system's `/tmp`. Commands that can write
  a large store check that room first; see
  [Writing outputs](../reference/commands.md#writing-outputs).

A scheduler's SIGTERM (a time limit, `scancel`, `bkill`) is handled like any
other failure: the temporary store is removed and the command exits with
code 143. Two runs writing the same output cannot overlap: the second is
refused while the first holds `out.h5ad.adata-lock`.

A run killed outright (by the out-of-memory killer, with SIGKILL) cannot
clean up after itself, and the next run that would write the same output
refuses to start:

```
Error: 'out.adata-tmp.h5ad' already exists, probably left by an interrupted
run. Check what it holds, remove it, and try again.
```

An `adata-tmp` store is always incomplete: delete it. An `adata-old` store is
the previous output, left because the run died between the two renames: if
`out.h5ad` is missing, rename `out.adata-old.h5ad` back; otherwise delete it.
A leftover `out.h5ad.adata-lock` goes too, once no run is writing.

{: .note }
In Nextflow, a retried task starts in a fresh work directory and never sees
leftovers. They arise when a script writes outside its work directory.

## Memory and `--chunk`

Memory follows `--chunk` rather than the size of the input. The defaults suit
most data:

| Commands | `--chunk` | Unit |
|---|---|---|
| `subset`, `split`, `concat`, `recast` | 1,024 | rows of a dense matrix |
| `export dataframe` | 10,000 | rows |
| `export array` | 100,000 | elements |

A dense block holds `--chunk` × n_var values, so lower it for very wide dense
matrices. It must be at least 1; zero or a negative value is a usage error. Sparse matrices stream in blocks too. obs and var columns are read
whole, which is what grows with the number of cells.
[Benchmarks](../reference/benchmarks.md) gives peak memory per command on a
50,000 × 20,000 store, measured on every release: between 70 and 330 MB in
0.16.1. A request of 1 to 2 GB leaves room for larger stores.

A loom output is dense by definition, so `convert` to `.loom` warns above
10 GB before compression.

## Exit codes and output streams

| Exit code | Meaning |
|---|---|
| 0 | Done. Warnings, if any, are on stderr |
| 1 | Refused or failed. One line on stderr starting `Error:` says why; nothing was written |
| 2 | A usage error: an unknown option, a missing argument, an input path that does not exist |
| 143 | Stopped by SIGTERM (a scheduler's time limit, `scancel`, `bkill`); the partial output was removed |

Data goes to stdout and everything else (progress, notes, warnings, errors)
to stderr, so `adata export dataframe in.h5ad obs > obs.csv` is clean, and
`2> adata.log` keeps the log. An export piped to a reader that stops early,
such as `head`, ends quietly with exit code 0.

## Zarr v2 and v3

Outputs keep their source's Zarr version, so a v2 store is never quietly
upgraded:

- A command on a Zarr store writes the same version.
- `concat` writes the version its Zarr inputs share; inputs of different
  versions give v3.
- A Zarr store written from an input that is not Zarr, such as an `.h5ad`, is v3.
- `--zarr-format 2` or `3` chooses, on `create`, `subset`, `split`, `concat`,
  `convert` and `recast`.

anndata reads Zarr v3 from 0.12, which needs zarr-python 3; for an older
reader downstream, write v2. A writable Zarr store's consolidated metadata is
rewritten when `adata` closes it, so anndata, which trusts that index, sees
the new members.
