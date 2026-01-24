"""
Tests for RustDataFrame.

These tests verify the Rust-backed DataFrame implementation.
Skip all tests if the Rust backend is not available.
"""
import pytest
import tempfile
import os

# Check if Rust backend is available
try:
    from nirvana_core import PyDataFrame
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    PyDataFrame = None

# Skip all tests if Rust backend not available
pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="Rust backend (nirvana_core) not available"
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    csv_content = """name,age,score
Alice,30,95.5
Bob,25,88.0
Charlie,35,92.0
"""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


class TestPyDataFrame:
    """Test the Rust PyDataFrame directly."""
    
    def test_from_dict(self):
        """Test creating DataFrame from dict."""
        df = PyDataFrame.from_dict({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        assert df.nrows == 3
        assert df.ncols == 2
        assert set(df.columns) == {"a", "b"}
    
    def test_from_dict_with_floats(self):
        """Test creating DataFrame with float values."""
        df = PyDataFrame.from_dict({
            "x": [1.5, 2.5, 3.5],
            "y": [4.0, 5.0, 6.0]
        })
        assert df.nrows == 3
        head = df.head(2)
        assert len(head) == 2
        assert head[0]["x"] == 1.5
    
    def test_from_csv(self, sample_csv):
        """Test loading DataFrame from CSV."""
        df = PyDataFrame.from_csv(sample_csv)
        assert df.nrows == 3
        assert "name" in df.columns
        assert "age" in df.columns
        assert "score" in df.columns
    
    def test_getitem_column(self):
        """Test column access."""
        df = PyDataFrame.from_dict({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        col_a = df["a"]
        assert len(col_a) == 3
        assert col_a[0] == 1
        assert col_a[1] == 2
    
    def test_contains(self):
        """Test column existence check."""
        df = PyDataFrame.from_dict({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        assert "a" in df
        assert "c" not in df
    
    def test_head_tail(self):
        """Test head and tail methods."""
        df = PyDataFrame.from_dict({
            "a": [1, 2, 3, 4, 5]
        })
        head = df.head(2)
        assert len(head) == 2
        assert head[0]["a"] == 1
        
        tail = df.tail(2)
        assert len(tail) == 2
        assert tail[-1]["a"] == 5
    
    def test_get_batch(self):
        """Test batch access for LLM prompting."""
        df = PyDataFrame.from_dict({
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "b", "c", "d", "e"]
        })
        batch = df.get_batch(["a", "b"], 1, 4)
        assert "a" in batch
        assert "b" in batch
        assert len(batch["a"]) == 3
        assert batch["a"] == [2, 3, 4]
        assert batch["b"] == ["b", "c", "d"]
    
    def test_select(self):
        """Test column selection."""
        df = PyDataFrame.from_dict({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [1.0, 2.0, 3.0]
        })
        selected = df.select(["a", "c"])
        assert selected.ncols == 2
        assert set(selected.columns) == {"a", "c"}
    
    def test_slice(self):
        """Test row slicing."""
        df = PyDataFrame.from_dict({
            "a": [1, 2, 3, 4, 5]
        })
        sliced = df.slice(1, 3)
        assert sliced.nrows == 3
        head = sliced.head(3)
        assert head[0]["a"] == 2
        assert head[2]["a"] == 4
    
    def test_to_dicts(self):
        """Test conversion to list of dicts."""
        df = PyDataFrame.from_dict({
            "a": [1, 2],
            "b": ["x", "y"]
        })
        dicts = df.to_dicts()
        assert len(dicts) == 2
        assert dicts[0] == {"a": 1, "b": "x"}
        assert dicts[1] == {"a": 2, "b": "y"}
    
    def test_schema_dict(self):
        """Test getting schema as dict."""
        df = PyDataFrame.from_dict({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"]
        })
        schema = df.schema_dict()
        assert "int_col" in schema
        assert "str_col" in schema
        # The types should be Arrow types
        assert "Int64" in schema["int_col"]
        assert "Utf8" in schema["str_col"]


class TestRustDataFrame:
    """Test the Python RustDataFrame wrapper."""
    
    @pytest.fixture
    def rust_dataframe(self):
        """Import RustDataFrame only if Rust is available."""
        from nirvana.dataframe.rust_frame import DataFrame
        return DataFrame
    
    def test_from_dict(self, rust_dataframe):
        """Test creating RustDataFrame from dict."""
        RustDataFrame = rust_dataframe
        df = RustDataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert df.nrows == 3
        assert df.ncols == 2
    
    def test_from_csv(self, rust_dataframe, sample_csv):
        """Test loading RustDataFrame from CSV."""
        RustDataFrame = rust_dataframe
        df = RustDataFrame.from_csv(sample_csv)
        assert df.nrows == 3
        assert "name" in df.columns
    
    def test_from_pyarrow(self, rust_dataframe):
        """Test loading RustDataFrame from PyArrow."""
        import pyarrow as pa
        
        RustDataFrame = rust_dataframe
        table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        df = RustDataFrame.from_pyarrow(table)
        assert df.nrows == 3
        assert set(df.columns) == {"x", "y"}
    
    def test_to_pyarrow(self, rust_dataframe):
        """Test exporting to PyArrow."""
        import pyarrow as pa
        
        RustDataFrame = rust_dataframe
        df = RustDataFrame({"a": [1, 2, 3]})
        table = df.to_pyarrow()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 3
    
    def test_get_batch(self, rust_dataframe):
        """Test batch access."""
        RustDataFrame = rust_dataframe
        df = RustDataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "b", "c", "d", "e"]
        })
        batch = df.get_batch(["a", "b"], 0, 3)
        assert len(batch["a"]) == 3
        assert batch["a"] == [1, 2, 3]
    
    def test_lineage_initialization(self, rust_dataframe):
        """Test that lineage is properly initialized."""
        RustDataFrame = rust_dataframe
        df = RustDataFrame({"text": ["hello", "world"]})
        
        # Check that leaf_node exists (from LineageMixin)
        assert hasattr(df, 'leaf_node')
        assert df.leaf_node is not None
        assert df.leaf_node.op_name == "scan"
    
    def test_select(self, rust_dataframe):
        """Test column selection returns new RustDataFrame."""
        RustDataFrame = rust_dataframe
        df = RustDataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        selected = df.select(["a", "c"])
        
        assert isinstance(selected, RustDataFrame)
        assert selected.ncols == 2
        assert set(selected.columns) == {"a", "c"}
