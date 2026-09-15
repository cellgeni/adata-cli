"""Tests for CLI commands."""

import pytest
import csv
from pathlib import Path
import h5py
import numpy as np
from typer.testing import CliRunner
from adata.cli import app
from adata.commands.info import show_info
from adata.commands.export import export_table
from rich.console import Console


runner = CliRunner()


class TestViewCommand:
    """Tests for info command."""

    def test_view_command_success(self, sample_h5ad_file):
        """Test info command on valid file."""
        result = runner.invoke(app, ["view", str(sample_h5ad_file)])
        assert result.exit_code == 0
        assert "5 × 4" in result.stdout

    def test_view_command_nonexistent_file(self):
        """Test info command on non-existent file."""
        result = runner.invoke(app, ["view", "nonexistent.h5ad"])
        assert result.exit_code != 0

    def test_view_function_direct(self, sample_h5ad_file):
        """Test show_info function directly."""
        console = Console(stderr=True)
        # Should not raise exception
        show_info(sample_h5ad_file, console)

    def test_view_tree_flag(self, sample_h5ad_file):
        """Test info command with --tree flag."""
        result = runner.invoke(app, ["view", "--tree", str(sample_h5ad_file)])
        assert result.exit_code == 0
        # Should show type annotations in angle brackets
        # Output may go to stdout or stderr depending on console config
        output = result.stdout + (result.stderr or "")
        assert "<" in output
        assert ">" in output

    def test_view_tree_short_flag(self, sample_h5ad_file):
        """Test info command with -t short flag."""
        result = runner.invoke(app, ["view", "-t", str(sample_h5ad_file)])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "<" in output

    def test_view_depth_flag(self, sample_h5ad_file):
        """Test info command with --depth flag."""
        result = runner.invoke(
            app, ["view", "--tree", "--depth", "1", str(sample_h5ad_file)]
        )
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "<" in output

    def test_view_depth_short_flag(self, sample_h5ad_file):
        """Test info command with -d short flag."""
        result = runner.invoke(app, ["view", "-t", "-d", "2", str(sample_h5ad_file)])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "<" in output

    def test_view_entry_positional(self, sample_h5ad_file):
        """Test info command with entry as positional argument."""
        result = runner.invoke(app, ["view", str(sample_h5ad_file), "X"])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "Path:" in output
        assert "Type:" in output

    def test_view_entry_obs(self, sample_h5ad_file):
        """Test info command with obs entry."""
        result = runner.invoke(app, ["view", str(sample_h5ad_file), "obs"])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "Path:" in output
        assert "dataframe" in output

    def test_view_entry_nested_path(self, sample_h5ad_file):
        """Test info command with nested object path."""
        result = runner.invoke(app, ["view", str(sample_h5ad_file), "uns/description"])
        assert result.exit_code == 0
        output = result.stdout + (result.stderr or "")
        assert "Path:" in output

    def test_view_entry_not_found(self, sample_h5ad_file):
        """Test info command with non-existent object path."""
        result = runner.invoke(app, ["view", str(sample_h5ad_file), "nonexistent"])
        assert result.exit_code == 0  # Doesn't exit with error, just shows message
        output = result.stdout + (result.stderr or "")
        assert "not found" in output


class TestExportDataframeCommand:
    """Tests for export dataframe command (replaces table command)."""

    def test_export_dataframe_obs(self, sample_h5ad_file, temp_dir):
        """Test export dataframe for obs axis."""
        output = temp_dir / "obs_table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

        # Check CSV content
        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 6  # header + 5 rows
            assert "obs_names" in rows[0]

    def test_export_dataframe_var(self, sample_h5ad_file, temp_dir):
        """Test export dataframe for var axis."""
        output = temp_dir / "var_table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "var",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 5  # header + 4 rows

    def test_export_dataframe_columns_filter(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with column filter."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
                "--columns",
                "obs_names,cell_type",
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = rows[0]
            assert "obs_names" in header
            assert "cell_type" in header
            assert "n_counts" not in header

    def test_export_dataframe_head(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with head limit."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
                "--head",
                "2",
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 3  # header + 2 rows

    def test_export_dataframe_head_short_flag(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with -n short flag."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
                "-n",
                "3",
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 4  # header + 3 rows

    def test_export_dataframe_stdout(self, sample_h5ad_file):
        """Test export dataframe to stdout (no --output)."""
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--head",
                "2",
            ],
        )
        assert result.exit_code == 0
        # Output should go to stdout
        assert "obs_names" in result.stdout
        assert "cell_" in result.stdout

    def test_export_dataframe_columns_short_flag(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with -c short flag for columns."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "-o",
                str(output),
                "-c",
                "obs_names",
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = rows[0]
            assert len(header) == 1
            assert "obs_names" in header

    def test_export_dataframe_chunk_rows(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with custom chunk size."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "--output",
                str(output),
                "--chunk-rows",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 6  # header + 5 rows

    def test_export_dataframe_combined_options(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with multiple options combined."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "obs",
                "-o",
                str(output),
                "-c",
                "obs_names,cell_type",
                "-n",
                "3",
                "-r",
                "1",
            ],
        )
        assert result.exit_code == 0

        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 4  # header + 3 rows
            header = rows[0]
            assert "obs_names" in header
            assert "cell_type" in header
            assert "n_counts" not in header

    def test_export_dataframe_multi_categorical_columns(self, temp_dir):
        """Test exporting multiple categorical columns keeps each column's values."""
        file_path = temp_dir / "categorical_obs.h5ad"

        with h5py.File(file_path, "w") as f:
            obs = f.create_group("obs")
            obs.attrs["_index"] = "_index"
            obs.create_dataset(
                "_index",
                data=np.array(["cell1", "cell2", "cell3", "cell4"], dtype="S"),
            )

            age = obs.create_group("age")
            age.attrs["encoding-type"] = "categorical"
            age.attrs["encoding-version"] = "0.2.0"
            age.attrs["ordered"] = np.False_
            age.create_dataset(
                "categories",
                data=np.array(["nan", "9.2", "11.0"], dtype=object),
            )
            age.create_dataset("codes", data=np.array([0, 1, 2, 0], dtype=np.int8))

            sample_id = obs.create_group("sample_ID")
            sample_id.attrs["encoding-type"] = "categorical"
            sample_id.attrs["encoding-version"] = "0.2.0"
            sample_id.attrs["ordered"] = np.False_
            sample_id.create_dataset(
                "categories",
                data=np.array(["nan", "XHU:307", "BRC2243"], dtype=object),
            )
            sample_id.create_dataset(
                "codes", data=np.array([0, 1, 2, 0], dtype=np.int8)
            )

            cell_type = obs.create_group("cell_type")
            cell_type.attrs["encoding-type"] = "categorical"
            cell_type.attrs["encoding-version"] = "0.2.0"
            cell_type.attrs["ordered"] = np.False_
            cell_type.create_dataset(
                "categories",
                data=np.array(
                    ["nan", "Forebrain neuronal IPC", "β-cell"], dtype=object
                ),
            )
            cell_type.create_dataset(
                "codes", data=np.array([0, 1, 2, 0], dtype=np.int8)
            )

            var = f.create_group("var")
            var.attrs["_index"] = "var_names"
            var.create_dataset("var_names", data=np.array(["g1", "g2"], dtype="S"))
            f.create_dataset("X", data=np.ones((4, 2), dtype=np.float32))

        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(file_path),
                "obs",
                "-c",
                "age,sample_ID,cell_type",
                "--head",
                "4",
            ],
        )

        assert result.exit_code == 0
        rows = list(csv.reader(result.stdout.splitlines()))
        assert rows[0] == ["_index", "age", "sample_ID", "cell_type"]
        assert rows[1] == ["cell1", "nan", "nan", "nan"]
        assert rows[2] == ["cell2", "9.2", "XHU:307", "Forebrain neuronal IPC"]
        assert rows[3] == ["cell3", "11.0", "BRC2243", "β-cell"]
        assert rows[4] == ["cell4", "nan", "nan", "nan"]

    def test_export_dataframe_missing_entry(self, sample_h5ad_file, temp_dir):
        """Test export dataframe with invalid axis."""
        output = temp_dir / "table.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "dataframe",
                str(sample_h5ad_file),
                "invalid",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 1
        # Check both stdout and stderr since Console uses stderr=True
        output_text = result.stdout + result.stderr
        assert "not found" in output_text

    def test_export_table_function(self, sample_h5ad_file, temp_dir):
        """Test export_table function directly."""
        output = temp_dir / "test_table.csv"
        console = Console(stderr=True)

        export_table(
            file=sample_h5ad_file,
            axis="obs",
            columns=None,
            out=output,
            chunk_rows=10,
            head=None,
            console=console,
        )

        assert output.exists()
        with open(output, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) > 1


class TestSubsetCommand:
    """Tests for subset command."""

    def test_subset_command_obs(self, sample_h5ad_file, temp_dir):
        """Test subset command with obs file."""
        obs_file = temp_dir / "obs.txt"
        obs_file.write_text("cell_1\ncell_3\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app,
            [
                "subset",
                str(sample_h5ad_file),
                "--output",
                str(output),
                "--obs",
                str(obs_file),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_subset_command_var(self, sample_h5ad_file, temp_dir):
        """Test subset command with var file."""
        var_file = temp_dir / "var.txt"
        var_file.write_text("gene_1\ngene_2\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app,
            [
                "subset",
                str(sample_h5ad_file),
                "--output",
                str(output),
                "--var",
                str(var_file),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_subset_command_both(self, sample_h5ad_file, temp_dir):
        """Test subset command with both obs and var files."""
        obs_file = temp_dir / "obs.txt"
        obs_file.write_text("cell_1\ncell_2\n")

        var_file = temp_dir / "var.txt"
        var_file.write_text("gene_1\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app,
            [
                "subset",
                str(sample_h5ad_file),
                "--output",
                str(output),
                "--obs",
                str(obs_file),
                "--var",
                str(var_file),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_subset_command_no_filters(self, sample_h5ad_file, temp_dir):
        """Test subset command without any filters (should fail)."""
        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app, ["subset", str(sample_h5ad_file), "--output", str(output)]
        )
        assert result.exit_code == 1
        # Check both stdout and stderr since Console uses stderr=True
        output_text = result.stdout + result.stderr
        assert "At least one of --obs or --var must be provided" in output_text

    def test_subset_command_chunk_rows(self, sample_h5ad_file, temp_dir):
        """Test subset command with custom chunk size."""
        obs_file = temp_dir / "obs.txt"
        obs_file.write_text("cell_1\ncell_2\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app,
            [
                "subset",
                str(sample_h5ad_file),
                "--output",
                str(output),
                "--obs",
                str(obs_file),
                "--chunk-rows",
                "512",
            ],
        )
        assert result.exit_code == 0
        assert output.exists()

    def test_subset_command_sparse(self, sample_sparse_csr_h5ad, temp_dir):
        """Test subset command on sparse matrix file."""
        obs_file = temp_dir / "obs.txt"
        obs_file.write_text("cell_1\ncell_3\n")

        output = temp_dir / "subset.h5ad"
        result = runner.invoke(
            app,
            [
                "subset",
                str(sample_sparse_csr_h5ad),
                "--output",
                str(output),
                "--obs",
                str(obs_file),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_help(self):
        """Test CLI help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Streaming CLI" in result.stdout

    def test_view_help(self):
        """Test view command help."""
        result = runner.invoke(app, ["view", "--help"])
        assert result.exit_code == 0
        assert "Show high-level information" in result.stdout

    def test_info_alias_is_deprecated(self):
        """The `info` alias still works but warns."""
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0
        assert "Deprecated alias" in result.stdout

    def test_export_help(self):
        """Test export command help."""
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "dataframe" in result.stdout
        assert "array" in result.stdout

    def test_export_dataframe_help(self):
        """Test export dataframe command help."""
        result = runner.invoke(app, ["export", "dataframe", "--help"])
        assert result.exit_code == 0
        assert "Export any dataframe" in result.stdout

    def test_import_help(self):
        """Test import command help."""
        result = runner.invoke(app, ["import", "--help"])
        assert result.exit_code == 0
        assert "dataframe" in result.stdout
        assert "array" in result.stdout

    def test_subset_help(self):
        """Test subset command help."""
        result = runner.invoke(app, ["subset", "--help"])
        assert result.exit_code == 0
        assert "Subset an AnnData store" in result.stdout
