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


def test_clamping_a_chunk_drops_an_incompatible_shard():
    from adata.core.subset import _clamp_chunks

    kept = _clamp_chunks({"chunks": (4, 2), "shards": (8, 4)}, 100, 100)
    assert kept["shards"] == (8, 4), "an unclamped chunk keeps its shard"

    dropped = _clamp_chunks({"chunks": (4, 2), "shards": (8, 4)}, 3, 4)
    assert dropped["chunks"] == (3, 2)
    assert "shards" not in dropped


def test_clamping_handles_one_dimensional_chunks():
    from adata.core.subset import _clamp_chunks

    assert _clamp_chunks({"chunks": (100,)}, 5)["chunks"] == (5,)
    assert _clamp_chunks({}, 5) == {}
