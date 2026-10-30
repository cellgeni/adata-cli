"""Tests for the storage layer: backend detection, copying, and Zarr versions.

This layer is what lets every other module stay backend-agnostic, so its job
is to make HDF5 and Zarr behave identically. The cases that matter are the
ones where they genuinely differ -- string dtypes, attribute types, codecs,
and Zarr's consolidated metadata index.
"""

from __future__ import annotations

import numpy as np
import pytest

from adata.elements import spec
from adata.elements import write as ew
from adata.storage import (
    Store,
    copy_attrs,
    copy_dataset,
    copy_path,
    copy_store_contents,
    copy_tree,
    dataset_create_kwargs,
    detect_backend,
    fit_chunks,
    has_valid_anndata_root_attrs,
    is_dataset,
    is_group,
    is_zarr_path,
    open_store,
    zarr_format_of,
)

zarr = pytest.importorskip("zarr")


# ---------------------------------------------------------------------------
# detection


def test_detect_backend_from_an_existing_file(temp_dir):
    path = temp_dir / "x.h5ad"
    with open_store(path, "w"):
        pass
    assert detect_backend(path) == "hdf5"


def test_detect_backend_from_an_existing_zarr_directory(temp_dir):
    path = temp_dir / "x.zarr"
    with open_store(path, "w"):
        pass
    assert detect_backend(path) == "zarr"
    assert is_zarr_path(path)


@pytest.mark.parametrize(
    "name,expected", [("new.zarr", "zarr"), ("new.h5ad", "hdf5"), ("new", "hdf5")]
)
def test_detect_backend_of_a_path_that_does_not_exist_yet(temp_dir, name, expected):
    assert detect_backend(temp_dir / name) == expected


def test_detect_backend_rejects_a_plain_directory(temp_dir):
    plain = temp_dir / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="does not look like a Zarr store"):
        detect_backend(plain)


def test_is_zarr_path_is_false_for_a_file(temp_dir):
    f = temp_dir / "f.txt"
    f.write_text("x")
    assert not is_zarr_path(f)


# ---------------------------------------------------------------------------
# root attributes


def test_writable_open_stamps_the_anndata_root(new_store):
    path, opener = new_store()
    with opener("r") as root:
        assert has_valid_anndata_root_attrs(root)


def test_reading_a_non_anndata_store_warns(temp_dir):
    import h5py

    path = temp_dir / "plain.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=[1])

    with pytest.warns(UserWarning, match="missing or invalid AnnData attrs"):
        with open_store(path, "r"):
            pass


def test_the_warning_can_be_suppressed_for_format_agnostic_commands(temp_dir):
    import h5py
    import warnings

    path = temp_dir / "plain.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=[1])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with open_store(path, "r", require_anndata=False):
            pass


# ---------------------------------------------------------------------------
# zarr versions


@pytest.mark.parametrize("version", [2, 3])
def test_requested_zarr_format_is_what_gets_written(temp_dir, version):
    path = temp_dir / f"v{version}.zarr"
    with open_store(path, "w", zarr_format=version) as store:
        assert store.zarr_format == version
    assert zarr_format_of(zarr.open_group(str(path))) == version


def test_v2_marker_files_are_written_for_v2(temp_dir):
    path = temp_dir / "v2.zarr"
    with open_store(path, "w", zarr_format=2):
        pass
    assert (path / ".zgroup").exists()


def test_hdf5_store_reports_no_zarr_format(temp_dir):
    with open_store(temp_dir / "x.h5ad", "w") as store:
        assert store.zarr_format is None


def test_writes_survive_a_consolidated_index(temp_dir):
    """Members added to a consolidated store must be visible on reopen."""
    path = temp_dir / "c.zarr"
    with open_store(path, "w") as store:
        ew.write_scalar(store.root, "first", 1)
    # Reopen and add another; the index from the first write is now stale.
    with open_store(path, "a") as store:
        ew.write_scalar(store.root, "second", 2)

    reopened = zarr.open_group(str(path))
    assert {"first", "second"} <= set(reopened.keys())


# ---------------------------------------------------------------------------
# copying


@pytest.mark.parametrize("src_backend", ["h5ad", "zarr2", "zarr3"])
@pytest.mark.parametrize("dst_backend", ["h5ad", "zarr2", "zarr3"])
def test_store_contents_copy_between_every_backend_pair(
    temp_dir, src_backend, dst_backend
):
    """Every crossing must carry text, numbers and structure intact.

    HDF5 reports a variable-length string dataset as `object`, which Zarr
    rejects, and Zarr's `<U` dtypes have no HDF5 equivalent -- so a copy that
    reuses the source dtype fails in both directions.
    """
    from tests.conftest import backend_path, backend_zarr_format

    src_path = backend_path(temp_dir, src_backend, "src")
    dst_path = backend_path(temp_dir, dst_backend, "dst")

    with open_store(
        src_path, "w", zarr_format=backend_zarr_format(src_backend)
    ) as store:
        root = store.root
        ew.write_dataframe_header(root, "obs", ["c1", "c2"], ["label"])
        ew.write_categorical(root["obs"], "label", [0, 1], ["x", "y"], ordered=True)
        ew.write_dataframe_header(root, "var", ["g1"], [])
        ew.ensure_anndata_skeleton(root)
        ew.write_string_array(root["uns"], "text", ["alpha", "café"])
        ew.write_dense(root["uns"], "nums", np.arange(6).reshape(2, 3))
        ew.write_sparse(root, "X", [1.0, 2.0], [0, 0], [0, 1, 2], (2, 1))

    with open_store(src_path, "r") as src, open_store(
        dst_path, "w", zarr_format=backend_zarr_format(dst_backend)
    ) as dst:
        copy_store_contents(src.root, dst.root)

    with open_store(dst_path, "r") as dst:
        root = dst.root
        from adata.elements.read import read_str_all

        assert read_str_all(root["uns"]["text"]) == ["alpha", "café"]
        assert np.array_equal(root["uns"]["nums"][...], np.arange(6).reshape(2, 3))
        assert spec.encoding_type(root["obs"]["label"]) == spec.CATEGORICAL
        assert spec.decode_attr(root["obs"]["label"].attrs["ordered"]) is True
        assert spec.encoding_type(root["X"]) == spec.CSR_MATRIX
        assert has_valid_anndata_root_attrs(root)


def _is_null_in(element, backend: str) -> bool:
    """Is `element` a null in the spelling `backend` uses for one?"""
    if spec.encoding_type(element) != spec.NULL:
        return False
    if backend == "h5ad":
        return element.shape is None
    return element.shape == () and element.dtype == bool


@pytest.mark.parametrize("src_backend", ["h5ad", "zarr2", "zarr3"])
@pytest.mark.parametrize("dst_backend", ["h5ad", "zarr2", "zarr3"])
def test_a_null_copies_between_every_backend_pair(temp_dir, src_backend, dst_backend):
    """anndata 0.13 writes a store without `raw` as `raw: null`.

    On HDF5 that is an empty dataspace, which has no value to read, so copying
    one to Zarr raised a TypeError -- and with it every conversion and every
    `subset -o out.zarr` of such a store.
    """
    from tests.conftest import backend_path, backend_zarr_format

    src_path = backend_path(temp_dir, src_backend, "src")
    dst_path = backend_path(temp_dir, dst_backend, "dst")
    with open_store(src_path, "w", zarr_format=backend_zarr_format(src_backend)) as store:
        ew.write_null(store.root, "raw")

    with open_store(src_path, "r") as src, open_store(
        dst_path, "w", zarr_format=backend_zarr_format(dst_backend)
    ) as dst:
        copy_store_contents(src.root, dst.root)

    with open_store(dst_path, "r") as dst:
        assert _is_null_in(dst.root["raw"], dst_backend)


@pytest.mark.parametrize("src_backend", ["h5ad", "zarr2", "zarr3"])
@pytest.mark.parametrize("dst_backend", ["h5ad", "zarr2", "zarr3"])
def test_an_empty_chunked_array_copies_between_every_backend_pair(
    temp_dir, src_backend, dst_backend
):
    """Zarr keeps a chunk larger than its array; h5py refuses to create one."""
    from tests.conftest import backend_path, backend_zarr_format

    src_path = backend_path(temp_dir, src_backend, "src")
    dst_path = backend_path(temp_dir, dst_backend, "dst")
    with open_store(src_path, "w", zarr_format=backend_zarr_format(src_backend)) as store:
        if src_backend == "h5ad":
            store.root.create_dataset(
                "empty", shape=(0,), maxshape=(None,), chunks=(4,), dtype="i4"
            )
        else:
            store.root.create_array("empty", shape=(0,), chunks=(4,), dtype="i4")

    with open_store(src_path, "r") as src, open_store(
        dst_path, "w", zarr_format=backend_zarr_format(dst_backend)
    ) as dst:
        copy_store_contents(src.root, dst.root)

    with open_store(dst_path, "r") as dst:
        assert dst.root["empty"].shape == (0,)


@pytest.mark.parametrize("version", [2, 3])
def test_compressed_zarr_stays_compressed_in_hdf5(temp_dir, version):
    """Zarr codecs cannot cross to HDF5, but leaving the copy uncompressed
    made it several times the size of its source."""
    src_path, dst_path = temp_dir / "src.zarr", temp_dir / "dst.h5ad"
    with open_store(src_path, "w", zarr_format=version) as store:
        packed = store.root.create_array("packed", shape=(1000,), chunks=(100,), dtype="f4")
        packed[:] = 1.0
        store.root.create_array(
            "plain", shape=(1000,), chunks=(100,), dtype="f4",
            compressors=None,
        )
        ew.write_string_array(store.root, "text", ["a", "b"])

    with open_store(src_path, "r") as src, open_store(dst_path, "w") as dst:
        copy_store_contents(src.root, dst.root)

    with open_store(dst_path, "r") as dst:
        assert dst.root["packed"].compression == "gzip"
        assert dst.root["plain"].compression is None, "an uncompressed source stays so"
        assert dst.root["text"].compression is None, "strings are not compressed"
        assert np.all(dst.root["packed"][...] == 1.0)


def test_copy_tree_can_exclude_members(new_store):
    path, opener = new_store()
    with opener("a") as root:
        group = ew.write_mapping(root["uns"], "src")
        ew.write_scalar(group, "keep", 1)
        ew.write_scalar(group, "drop", 2)
        copy_tree(group, root["uns"], "dst", exclude={"drop"})
        assert "keep" in root["uns"]["dst"]
        assert "drop" not in root["uns"]["dst"]


def test_copy_tree_rejects_an_unsupported_object(new_store):
    path, opener = new_store()
    with opener("a") as root:
        with pytest.raises(TypeError, match="Unsupported object type"):
            copy_tree(object(), root["uns"], "x")


def test_copy_dataset_preserves_a_scalar(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_scalar(root["uns"], "s", 5)
        copy_dataset(root["uns"]["s"], root["uns"], "copy")
        assert root["uns"]["copy"][()] == 5


def test_copy_attrs_makes_values_json_safe_for_zarr(temp_dir):
    """Zarr attributes must be JSON; a numpy array there raises."""
    import h5py

    src_path = temp_dir / "src.h5ad"
    with h5py.File(src_path, "w") as f:
        f.attrs["arr"] = np.array([1, 2, 3])
        f.attrs["text"] = b"bytes"

    dst_path = temp_dir / "dst.zarr"
    with h5py.File(src_path, "r") as src, open_store(dst_path, "w") as dst:
        copy_attrs(src.attrs, dst.root.attrs, target_backend="zarr")

    with open_store(dst_path, "r") as dst:
        assert list(dst.root.attrs["arr"]) == [1, 2, 3]
        assert dst.root.attrs["text"] == "bytes"


def test_dataset_create_kwargs_carries_chunking_from_a_chunked_source(new_store):
    path, opener = new_store()
    with opener("a") as root:
        from adata.storage import create_dataset

        create_dataset(
            root["uns"], "a", shape=(10, 10), dtype="int64", chunks=(5, 5)
        )
        kwargs = dataset_create_kwargs(root["uns"]["a"], target_backend="hdf5")
        assert tuple(kwargs["chunks"]) == (5, 5)


def test_dataset_create_kwargs_drops_codecs_across_zarr_versions(temp_dir):
    """v2 numcodecs and v3 codec classes are not interchangeable."""
    src = temp_dir / "v3.zarr"
    with open_store(src, "w", zarr_format=3) as store:
        ew.write_dense(store.root, "a", np.arange(100).reshape(10, 10))

    with open_store(src, "r") as store:
        crossing = dataset_create_kwargs(
            store.root["a"], target_backend="zarr", zarr_format=2
        )
        staying = dataset_create_kwargs(
            store.root["a"], target_backend="zarr", zarr_format=3
        )

    assert "compressors" not in crossing and "compressor" not in crossing
    assert "compressors" in staying


def test_copy_path_duplicates_a_store_on_disk(temp_dir):
    src = temp_dir / "src.zarr"
    with open_store(src, "w") as store:
        ew.write_scalar(store.root, "x", 1)

    dst = temp_dir / "dst.zarr"
    copy_path(src, dst)
    with open_store(dst, "r") as store:
        assert store.root["x"][()] == 1


def test_copy_path_refuses_to_overwrite_a_zarr_store(temp_dir):
    src = temp_dir / "src.zarr"
    dst = temp_dir / "dst.zarr"
    for path in (src, dst):
        with open_store(path, "w"):
            pass
    with pytest.raises(FileExistsError):
        copy_path(src, dst)


# ---------------------------------------------------------------------------
# predicates


def test_group_and_dataset_predicates_agree_across_backends(new_store):
    path, opener = new_store()
    with opener("a") as root:
        ew.write_dense(root["uns"], "d", np.zeros(3))
        assert is_group(root["uns"]) and not is_dataset(root["uns"])
        assert is_dataset(root["uns"]["d"]) and not is_group(root["uns"]["d"])


def test_store_close_is_safe_to_call_twice(temp_dir):
    store = open_store(temp_dir / "x.h5ad", "w")
    store.close()
    store.close()


# ---------------------------------------------------------------------------
# regressions from review of #9


def test_target_format_is_taken_from_the_destination(temp_dir):
    """Callers that forget to pass a format must still get correct kwargs.

    Defaulting an unknown target to v3 meant v3-only options -- sharding
    especially -- were forwarded into v2 arrays, which reject them.
    """
    src_path = temp_dir / "v3.zarr"
    with open_store(src_path, "w", zarr_format=3) as store:
        ew.write_dense(store.root, "a", np.arange(64).reshape(8, 8))

    dst_path = temp_dir / "v2.zarr"
    with open_store(src_path, "r") as src, open_store(
        dst_path, "w", zarr_format=2
    ) as dst:
        kwargs = dataset_create_kwargs(
            src.root["a"], target_backend="zarr", dst_parent=dst.root
        )
        assert "shards" not in kwargs
        assert "compressors" not in kwargs and "compressor" not in kwargs

        # With no destination and no format, nothing version-specific travels.
        blind = dataset_create_kwargs(src.root["a"], target_backend="zarr")
        assert "shards" not in blind


@pytest.mark.parametrize("target", [2, 3])
def test_subsetting_a_sharded_v3_store(temp_dir, target):
    """A shard must survive or be dropped, never break array creation.

    Zarr requires a shard to be a whole number of chunks, so clamping the
    chunk to the subset size invalidates the source's shard geometry.
    """
    src_path = temp_dir / "src.zarr"
    with open_store(src_path, "w", zarr_format=3) as store:
        root = store.root
        ew.write_dataframe_header(root, "obs", [f"c{i}" for i in range(8)], [])
        ew.write_dataframe_header(root, "var", [f"g{i}" for i in range(4)], [])
        ew.ensure_anndata_skeleton(root)
        from adata.storage import create_dataset

        matrix = create_dataset(
            root, "X", shape=(8, 4), dtype="float32",
            chunks=(4, 2), shards=(8, 4),
        )
        matrix[...] = np.arange(32, dtype="float32").reshape(8, 4)
        spec.set_encoding(matrix, spec.ARRAY)

    names = temp_dir / "keep.txt"
    names.write_text("c0\nc2\nc4\n")
    out = temp_dir / f"out{target}.zarr"

    from adata.core.subset import subset_h5ad
    from rich.console import Console

    subset_h5ad(
        file=src_path, output=out, obs_file=names, var_file=None,
        console=Console(stderr=True), zarr_format=target,
    )

    with open_store(out, "r") as store:
        assert store.zarr_format == target
        assert store.root["X"].shape == (3, 4)
        expected = np.arange(32, dtype="float32").reshape(8, 4)[[0, 2, 4]]
        assert np.array_equal(store.root["X"][...], expected)


def test_fitting_a_chunk_drops_an_incompatible_shard():
    kept = fit_chunks({"chunks": (4, 2), "shards": (8, 4)}, (100, 100), backend="zarr")
    assert kept["shards"] == (8, 4), "an unclamped chunk keeps its shard"

    dropped = fit_chunks({"chunks": (4, 2), "shards": (8, 4)}, (3, 4), backend="zarr")
    assert dropped["chunks"] == (3, 2)
    assert "shards" not in dropped


def test_fitting_handles_one_dimensional_chunks():
    assert fit_chunks({"chunks": (100,)}, (5,), backend="hdf5")["chunks"] == (5,)
    assert fit_chunks({}, (5,), backend="hdf5") == {}


def test_fitting_leaves_a_zero_length_hdf5_dataset_to_choose_its_chunks():
    """No chunk fits a zero-length dimension, and h5py rejects any it is given.

    Zarr accepts a chunk larger than its array, so it keeps the source's.
    """
    kw = {"chunks": (1,), "compression": "gzip"}
    assert fit_chunks(kw, (0,), backend="hdf5") == {"compression": "gzip"}
    assert fit_chunks(kw, (0,), backend="zarr") == kw


# ---------------------------------------------------------------------------
# read sizing


def test_read_step_grows_past_a_pathological_source_chunk():
    """A `(1, n_cols)` source must not be copied one row per read.

    Forwarding the source's chunk height meant inheriting whatever the writer
    chose. On a network filesystem each read is a round-trip, so a row-chunked
    million-cell store spent the whole copy waiting on latency.
    """
    from adata.storage import TARGET_READ_BYTES, _chunk_step, _row_bytes

    shape, chunks, dtype = (1_000_000, 30_000), (1, 30_000), np.dtype("float32")
    step = _chunk_step(shape, chunks, dtype)

    assert step > 1, "the source chunk height is no longer used verbatim"
    assert step * _row_bytes(shape, dtype) >= TARGET_READ_BYTES // 2


def test_read_step_stays_a_whole_number_of_source_chunks():
    """Reading part of a chunk still costs decompressing all of it."""
    from adata.storage import _chunk_step

    for chunk_rows in (1, 7, 100, 65_536):
        step = _chunk_step((10_000_000,), (chunk_rows,), np.dtype("int64"))
        assert step % chunk_rows == 0


def test_read_step_never_exceeds_the_dataset():
    from adata.storage import _chunk_step

    assert _chunk_step((10,), (1,), np.dtype("int64")) == 10
    assert _chunk_step((), None, np.dtype("int64")) == 1
    assert _chunk_step((0,), None, np.dtype("int64")) == 1


def test_read_step_bounds_peak_memory():
    """A wide unchunked source used to read 1024 rows however wide they were."""
    from adata.storage import TARGET_READ_BYTES, _chunk_step, _row_bytes

    cases = [
        ((1_000_000, 30_000), None, np.dtype("float32")),
        ((1_000_000, 50), (1, 50), np.dtype("float32")),
        ((200_000_000,), (65_536,), np.dtype("int64")),
    ]
    for shape, chunks, dtype in cases:
        step = _chunk_step(shape, chunks, dtype)
        assert step * _row_bytes(shape, dtype) <= 2 * TARGET_READ_BYTES


def test_row_bytes_does_not_trust_a_vlen_itemsize():
    """h5py reports itemsize 8 for vlen str -- that is the pointer, not the text."""
    import h5py

    from adata.storage import VLEN_ELEMENT_BYTES, _row_bytes

    assert _row_bytes((10,), h5py.string_dtype(encoding="utf-8")) == VLEN_ELEMENT_BYTES
    assert _row_bytes((10,), np.dtype("O")) == VLEN_ELEMENT_BYTES
    assert _row_bytes((10, 4), np.dtype("float32")) == 16


def test_copy_dataset_of_a_row_chunked_source_round_trips(new_store):
    """The larger read step must not change what lands on disk."""
    path, opener = new_store()
    with opener("a") as root:
        from adata.storage import create_dataset

        values = np.arange(400, dtype="float32").reshape(100, 4)
        create_dataset(root["uns"], "src", data=values, chunks=(1, 4))
        copy_dataset(root["uns"]["src"], root["uns"], "dst")
        assert np.array_equal(root["uns"]["dst"][...], values)


# ---------------------------------------------------------------------------
# Data held outside the file


def _anndata_root(f) -> None:
    import h5py

    f.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    for key, n in (("obs", 1), ("var", 8)):
        frame = ew.write_dataframe_header(f, key, [f"{key}{i}" for i in range(n)], [])
        assert isinstance(frame, h5py.Group)


def _outside_file(temp_dir, kind: str):
    """An h5ad whose X takes its data from another file."""
    import h5py

    other = temp_dir / "other.bin"
    other.write_bytes(bytes(range(8)))
    path = temp_dir / f"{kind}.h5ad"
    with h5py.File(temp_dir / "other.h5", "w") as f:
        f.create_dataset("data", data=np.arange(8, dtype="u1").reshape(1, 8))
    with h5py.File(path, "w") as f:
        _anndata_root(f)
        if kind == "external":
            ds = f.create_dataset("X", shape=(1, 8), dtype="u1", external=[(str(other), 0, 8)])
        elif kind == "virtual":
            layout = h5py.VirtualLayout(shape=(1, 8), dtype="u1")
            layout[:] = h5py.VirtualSource(str(temp_dir / "other.h5"), "data", shape=(1, 8))
            ds = f.create_virtual_dataset("X", layout)
        else:
            f["X"] = h5py.ExternalLink(str(temp_dir / "other.h5"), "data")
            return path
        spec.set_encoding(ds, spec.ARRAY)
    return path


KINDS = ["external", "virtual", "link"]


def _cli_out(result) -> str:
    import re

    text = result.stdout + (result.stderr or "")
    return " ".join(re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text).split())


@pytest.mark.parametrize("kind", KINDS)
def test_a_store_reading_another_file_is_refused_on_opening(temp_dir, kind):
    """F041: a dataset or link whose data lives in another file was read
    like any other, and copied into the output."""
    from adata.storage import ExternalDataError

    path = _outside_file(temp_dir, kind)
    with pytest.raises(ExternalDataError, match=r"/X (keeps its data|is a virtual|is an external link)"):
        open_store(path, "r")
    with pytest.raises(ExternalDataError):
        open_store(path, "a")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "command",
    [
        ["convert", "{src}", "-o", "{tmp}/out.zarr"],
        ["convert", "{src}", "-o", "{tmp}/out.h5ad"],
        ["subset", "{src}", "-q", "_index == 'obs0'", "-o", "{tmp}/out.h5ad"],
        ["export", "array", "{src}", "X", "-o", "{tmp}/out.npy"],
        ["view", "{src}"],
    ],
    ids=["convert-zarr", "convert-h5ad", "subset", "export", "view"],
)
def test_every_command_refuses_a_store_reading_another_file(temp_dir, kind, command):
    """F041: commands read, copied and exported such data without a word."""
    from typer.testing import CliRunner

    from adata.cli import app

    src = _outside_file(temp_dir, kind)
    args = [a.format(src=src, tmp=temp_dir) for a in command]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, _cli_out(result)
    assert "is refused" in _cli_out(result)
    assert not list(temp_dir.glob("out*"))


def test_an_hdf5_matrix_reading_another_file_is_refused_by_convert(temp_dir):
    """F041: the other converters open their HDF5 with h5py directly."""
    import h5py
    from typer.testing import CliRunner

    from adata.cli import app
    from tests.conftest import write_10x_h5

    h5 = write_10x_h5(
        temp_dir / "m.h5", np.eye(3, dtype=np.int32),
        barcodes=["a-1", "b-1", "c-1"], ids=["g1", "g2", "g3"], names=["G1", "G2", "G3"],
    )
    secret = temp_dir / "other.bin"
    secret.write_bytes(np.ones(3, dtype=np.int32).tobytes())
    with h5py.File(h5, "a") as f:
        del f["matrix/data"]
        f.create_dataset("matrix/data", shape=(3,), dtype=np.int32, external=[(str(secret), 0, 12)])
    result = CliRunner().invoke(app, ["convert", str(h5), "-o", str(temp_dir / "out.h5ad")])
    assert result.exit_code == 1
    assert "/matrix/data keeps its data in external files" in _cli_out(result)
    assert not (temp_dir / "out.h5ad").exists()


@pytest.mark.parametrize("kind", ["external", "virtual"])
def test_copy_tree_refuses_data_held_in_another_file(temp_dir, kind):
    """F041: a copy from a file not opened through open_store (MuData's
    modalities are copied from an h5py handle) is checked too."""
    import h5py

    from adata.storage import ExternalDataError

    src = _outside_file(temp_dir, kind)
    with h5py.File(src, "r") as f, open_store(temp_dir / "out.h5ad", "w") as dst:
        for compression in (None, "gzip"):
            with pytest.raises(ExternalDataError):
                copy_tree(f["X"], dst.root, f"X_{compression}", compression=compression)
        group = dst.root.create_group("holder")
        with pytest.raises(ExternalDataError):
            copy_tree(f, group, "everything")


def test_a_store_holding_only_its_own_data_opens(temp_dir):
    import h5py

    path = temp_dir / "plain.h5ad"
    with h5py.File(path, "w") as f:
        _anndata_root(f)
        f["X"] = np.ones((1, 8))
        f["alias"] = h5py.SoftLink("/X")
    with open_store(path, "r") as store:
        assert store.root["alias"].shape == (1, 8)


# ---------------------------------------------------------------------------
# Fill values and attributes


def test_a_fill_value_is_converted_with_the_dtype_or_left_out():
    """F071: a float source's fill went to a bool dataset unconverted, which
    h5py refuses; one the conversion would change is left to the default."""
    from adata.storage import _fill_as

    assert _fill_as(np.float32(1.0), np.bool_) == np.True_
    assert _fill_as(np.float64(7.0), np.int16) == 7
    assert _fill_as(np.float64(0.5), np.int16) is None
    assert _fill_as(np.float64(np.nan), np.int64) is None
    assert np.isnan(_fill_as(np.float32(np.nan), np.float64))
    assert _fill_as(np.int8(3), np.dtype([("a", "i4")])) is None


def test_only_a_numeric_fill_other_than_zero_is_carried(temp_dir):
    """F021: a record's fill, a value for each field, crashed h5py when the
    fields were text; and a zero fill is HDF5's default anyway."""
    import h5py

    path = temp_dir / "src.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("zero", shape=(4,), dtype="f4")
        f.create_dataset("nan", shape=(4,), dtype="f4", fillvalue=np.nan)
        f.create_dataset("records", shape=(4,), dtype=[("a", h5py.string_dtype()), ("b", "f4")])
    with h5py.File(path, "r") as f:
        assert "fillvalue" not in dataset_create_kwargs(f["zero"], target_backend="hdf5")
        assert np.isnan(dataset_create_kwargs(f["nan"], target_backend="hdf5")["fillvalue"])
        assert "fillvalue" not in dataset_create_kwargs(f["records"], target_backend="hdf5")
        with open_store(temp_dir / "dst.h5ad", "w") as dst:
            ds = copy_dataset(f["nan"], dst.root, "nan")
            assert np.isnan(ds.fillvalue)
            from adata.storage import create_dataset

            kw = dataset_create_kwargs(f["nan"], target_backend="hdf5")
            assert create_dataset(dst.root, "as_int", shape=(4,), dtype="i4", **kw).fillvalue == 0


def test_copy_attrs_decodes_nested_bytes_and_empties_for_zarr(temp_dir):
    """F130: bytes were decoded one level deep, and h5py.Empty not at all."""
    import h5py

    src_path = temp_dir / "src.h5ad"
    with h5py.File(src_path, "w") as f:
        f.attrs["grid"] = np.array([[b"a", b"b"], [b"c", b"d"]])
        f.attrs["nothing"] = h5py.Empty("f4")
        f.attrs["words"] = np.array(["x", "y"], dtype=h5py.string_dtype())
    with h5py.File(src_path, "r") as src, open_store(temp_dir / "dst.zarr", "w") as dst:
        copy_attrs(src.attrs, dst.root.attrs, target_backend="zarr")
    with open_store(temp_dir / "dst.zarr", "r") as dst:
        assert dst.root.attrs["grid"] == [["a", "b"], ["c", "d"]]
        assert dst.root.attrs["nothing"] is None
        assert dst.root.attrs["words"] == ["x", "y"]


def test_dataset_create_kwargs_keeps_a_v2_arrays_compression_or_its_absence(temp_dir):
    """F134, F135: a v2 array's codec was read through the deprecated
    `compressor`, and an uncompressed one forwarded nothing, so zarr gave it
    Blosc."""
    import numcodecs

    src = temp_dir / "v2.zarr"
    with open_store(src, "w", zarr_format=2) as store:
        store.root.create_array("zstd", data=np.arange(10), compressors=numcodecs.Zstd(level=5))
        store.root.create_array("none", data=np.arange(10), compressors=None)
    with open_store(src, "r") as store, open_store(temp_dir / "out.zarr", "w", zarr_format=2) as out:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            for key in ("zstd", "none"):
                kw = dataset_create_kwargs(store.root[key], target_backend="zarr", zarr_format=2)
                copy_dataset(store.root[key], out.root, key)
        assert out.root["zstd"].compressors == (numcodecs.Zstd(level=5),)
        assert out.root["none"].compressors == ()
        assert kw["compressors"] is None


def test_a_reference_to_an_object_no_link_names_is_refused(temp_dir):
    """F041: the check walked links, and an object reachable only through a
    reference -- here a legacy categorical's labels, created without a name
    and holding external data -- was never seen, and was read."""
    import h5py

    from adata.storage import ExternalDataError

    other = temp_dir / "other.bin"
    other.write_bytes(b"ABC")
    path = temp_dir / "anon.h5ad"
    with h5py.File(path, "w") as f:
        for key in ("obs", "var"):
            frame = f.create_group(key)
            frame.attrs.update({"_index": "_index", "encoding-type": "dataframe", "encoding-version": "0.1.0"})
            frame.create_dataset("_index", data=np.array([b"a", b"b", b"c"]))
        # A group linked to itself outlives its last name: what it holds is
        # then reachable by reference only.
        hidden = f.create_group("hidden")
        h5py.h5o.link(hidden.id, hidden.id, b"self")
        labels = hidden.create_dataset("labels", shape=(3,), dtype="S1", external=[(str(other), 0, 3)])
        codes = f["obs"].create_dataset("kind", data=np.array([0, 1, 2], dtype="i1"))
        codes.attrs["categories"] = labels.ref
        del f["hidden"]
    with pytest.raises(ExternalDataError, match="/obs/kind refers to an object with no name"):
        open_store(path, "r")
    with h5py.File(path, "r") as f:
        from adata.elements.read import categories_dataset

        with pytest.raises(ValueError, match="no name in the file"):
            categories_dataset(f["obs/kind"], f["obs"])


def test_the_hdf5_files_of_a_folder_are_checked_through_linked_folders(temp_dir):
    """F041: the folder walk did not follow a linked subfolder, which the
    converters' own globs do."""
    import h5py

    from adata.storage import ExternalDataError, check_hdf5_inputs

    outs = temp_dir / "outs"
    real = temp_dir / "elsewhere" / "binned"
    real.mkdir(parents=True)
    outs.mkdir()
    (outs / "binned_outputs").symlink_to(real, target_is_directory=True)
    (real / "loop").symlink_to(outs, target_is_directory=True)
    other = temp_dir / "other.bin"
    other.write_bytes(bytes(4))
    with h5py.File(real / "matrix.h5", "w") as f:
        f.create_dataset("data", shape=(4,), dtype="u1", external=[(str(other), 0, 4)])
    with pytest.raises(ExternalDataError, match="matrix.h5' is refused: /data keeps its data"):
        check_hdf5_inputs(outs)
