"""
DataFrame backed by Rust columnar storage.

This class provides DataFrame APIs compatible to the one in previous versions,
and uses arrow-rs as the columnar storage manager for improved performance.
"""
from typing import Callable, Literal, Any
import pyarrow as pa

from nirvana.lineage.mixin import LineageMixin

# Import Rust bindings (optional, with fallback)
try:
    from nirvana.nirvana_core import (
        PyDataFrame,
        PyColumn,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    PyDataFrame = None
    PyColumn = None


def rust_available() -> bool:
    """Check if the Rust backend is available."""
    return RUST_AVAILABLE


class DataFrame(LineageMixin):
    """
    DataFrame backed by Rust columnar storage (arrow-rs) for high-performance
    in-memory semantic data analytics.
    
    Supports:
    - Loading data from CSV, Excel, PyArrow, and Python dicts
    - Building data lineage with semantic operations
    - Query optimization as standard DataFrame does
    - Batch access for efficient LLM prompting
    
    Example:
        >>> df = DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        >>> df.nrows
        2
        >>> df.columns
        ['name', 'age']
        
        # Load from CSV
        >>> df = DataFrame.from_csv("data.csv")
        
        # Load from PyArrow
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> df = DataFrame.from_pyarrow(table)
    """
    
    def __init__(
        self,
        data: dict | pa.Table | None = None,
        *args,
        **kwargs
    ):
        """
        Create a new DataFrame.
        
        Args:
            data: Initial data as a dict, PyArrow Table, or None for empty DataFrame
            infer_media_types: If True, automatically detect and convert image/audio columns
        """
        if not RUST_AVAILABLE:
            raise ImportError(
                "Rust backend not available. Install nirvana-core with:\n"
                "  cd nirvana-core && maturin develop"
            )
        
        # Track column media types (column_name -> "image" | "audio" | "text")
        self._column_media_types: dict[str, str] = {}
        
        if isinstance(data, dict):
            # Pass data directly to Rust - it handles inference and conversion
            self._data = PyDataFrame.from_dict(data)
        elif isinstance(data, pa.Table):
            self._data = PyDataFrame.from_pyarrow(data)
        elif data is None:
            self._data = PyDataFrame()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        
        # Create a PyArrow view for pandas compatibility when needed
        self._arrow_table = None        
        self.initialize()

    @property
    def dtypes(self) -> dict[str, Any]:
        data_types = self._data.get_dtypes()
        return data_types 
    
    @classmethod
    def from_csv(
        cls,
        path: str,
        delimiter: str = ",",
        has_header: bool = True,
    ) -> "DataFrame":
        """
        Load DataFrame from a CSV file.
        
        Args:
            path: Path to the CSV file
            delimiter: Column delimiter character (default: ",")
            has_header: Whether the file has a header row (default: True)
            
        Returns:
            A new DataFrame with the CSV data
        """
        if not RUST_AVAILABLE:
            raise ImportError("Rust backend not available")
        
        instance = cls.__new__(cls)
        instance._data = PyDataFrame.from_csv(
            path, 
            ord(delimiter), 
            has_header
        )
        instance._arrow_table = None
        instance.initialize()
        return instance
    
    @classmethod
    def from_excel(
        cls,
        path: str,
        sheet: str | None = None,
    ) -> "DataFrame":
        """
        Load DataFrame from an Excel file.
        
        Args:
            path: Path to the Excel file (.xlsx)
            sheet: Optional sheet name (default: first sheet)
            
        Returns:
            A new DataFrame with the Excel data
        """
        if not RUST_AVAILABLE:
            raise ImportError("Rust backend not available")
        
        instance = cls.__new__(cls)
        instance._data = PyDataFrame.from_excel(path, sheet)
        instance._arrow_table = None
        instance.initialize()
        return instance
    
    @classmethod
    def from_pyarrow(cls, table: pa.Table) -> "DataFrame":
        """
        Load DataFrame from a PyArrow Table.
        
        Args:
            table: A PyArrow Table
            
        Returns:
            A new DataFrame
        """
        if not RUST_AVAILABLE:
            raise ImportError("Rust backend not available")
        
        instance = cls.__new__(cls)
        instance._data = PyDataFrame.from_pyarrow(table)
        instance._arrow_table = table
        instance.initialize()
        return instance
    
    @classmethod
    def from_external_file(cls, path: str, sep: str = ",", **kwargs) -> "DataFrame":
        """
        Load DataFrame from an external file (CSV or Excel based on extension).
        
        Args:
            path: Path to the file
            sep: Separator for CSV files (default: ",")
            **kwargs: Additional arguments
            
        Returns:
            A new DataFrame
        """
        if path.endswith(('.xlsx', '.xls')):
            return cls.from_excel(path, sheet=kwargs.get('sheet'))
        else:
            return cls.from_csv(path, delimiter=sep, has_header=kwargs.get('header', True))

    def __len__(self) -> int:
        return self.nrows
    
    def __contains__(self, item: str) -> bool:
        return item in self._data
    
    def __repr__(self) -> str:
        media_info = ""
        if self._column_media_types:
            media_cols = [f"{k}:{v}" for k, v in self._column_media_types.items() if v != "text"]
            if media_cols:
                media_info = f", media_columns=[{', '.join(media_cols)}]"
        return f"DataFrame(nrows={self.nrows}, ncols={len(self.columns)}{media_info})"
    
    @property
    def column_media_types(self) -> dict[str, str]:
        """Get column media types (column_name -> 'image' | 'audio' | 'text')."""
        return self._column_media_types.copy()
    
    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return self._data.columns
    
    @property
    def nrows(self) -> int:
        """Get number of rows."""
        return self._data.nrows
    
    @property
    def ncols(self) -> int:
        """Get number of columns."""
        return self._data.ncols
    
    @property
    def data(self):
        """
        Get pandas-compatible view of the data.
        This property exists for compatibility with LineageMixin.
        """
        # Lazily convert to pandas via PyArrow when needed
        return self.to_pyarrow().to_pandas()
    
    def head(self, n: int = 5) -> list[dict]:
        """Get first n rows as a list of dicts."""
        return self._data.head(n)
    
    def tail(self, n: int = 5) -> list[dict]:
        """Get last n rows as a list of dicts."""
        return self._data.tail(n)
    
    def to_pyarrow(self) -> pa.Table:
        """Convert to a PyArrow Table."""
        if self._arrow_table is None:
            self._arrow_table = self._data.to_pyarrow()
        return self._arrow_table
    
    def to_pandas(self):
        """Convert to a pandas DataFrame (requires pandas)."""
        return self.to_pyarrow().to_pandas()
    
    def to_dicts(self) -> list[dict]:
        """Convert to a list of dictionaries."""
        return self._data.to_dicts()
    
    def schema_dict(self) -> dict[str, str]:
        """Get schema as a dictionary of {column_name: dtype}."""
        return self._data.schema_dict()
    
    def _get(self, posidx, materialize: bool = False) -> Any:
        """
        Get data by position index.
        
        Args:
            posidx: Position index (int, str, slice, list, or tuple)
            materialize: Whether to materialize the result
            
        Returns:
            Column data or row data depending on index type
        """
        if isinstance(posidx, str):
            # str index => column selection
            if posidx in self.columns:
                return self._data[posidx]
            raise KeyError(f"Column `{posidx}` does not exist.")

        elif isinstance(posidx, int):
            # int index => single row as dict
            rows = self._data.head(posidx + 1) if posidx >= 0 else self._data.tail(-posidx)
            if posidx >= 0:
                return rows[posidx] if posidx < len(rows) else None
            else:
                return rows[0] if rows else None
        
        index_type = None
        if isinstance(posidx, slice):
            index_type = "row"
        elif (isinstance(posidx, tuple) or isinstance(posidx, list)) and len(posidx):
            if isinstance(posidx[0], str):
                index_type = "column"
            else:
                index_type = "row"
        else:
            raise TypeError(f"Invalid index type: {type(posidx)}")
        
        if index_type == "row":
            # For row slicing, use slice method
            if isinstance(posidx, slice):
                start = posidx.start or 0
                stop = posidx.stop or self.nrows
                length = stop - start
                sliced = self._data.slice(start, length)
                return sliced.to_dicts()
            else:
                # List of row indices - return as dicts
                return [self._get(i) for i in posidx]
        elif index_type == "column":
            # Select specific columns
            return self._data.select(list(posidx))

    def __getitem__(self, posidx):
        return self._get(posidx)
    
    def get_batch(
        self, 
        columns: list[str], 
        start: int, 
        end: int
    ) -> dict[str, list]:
        """
        Get a batch of data across specified columns.
        
        This is optimized for LLM prompting where you need to pack
        multiple column values together for each row in a batch.
        
        Args:
            columns: List of column names to include
            start: Start row index (inclusive)
            end: End row index (exclusive)
            
        Returns:
            Dictionary mapping column names to lists of values
            
        Example:
            >>> df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            >>> df.get_batch(["a", "b"], 0, 2)
            {'a': [1, 2], 'b': ['x', 'y']}
        """
        return self._data.get_batch(columns, start, end)
    
    def select(self, columns: list[str]) -> "DataFrame":
        """
        Select specific columns, returning a new DataFrame.
        
        Args:
            columns: List of column names to select
            
        Returns:
            A new DataFrame with only the specified columns
        """
        instance = DataFrame.__new__(DataFrame)
        instance._data = self._data.select(columns)
        instance._arrow_table = None
        instance.initialize()
        return instance
    
    def slice(self, start: int, length: int) -> "DataFrame":
        """
        Slice rows by range.
        
        Args:
            start: Start row index
            length: Number of rows to include
            
        Returns:
            A new DataFrame with the sliced rows
        """
        instance = DataFrame.__new__(DataFrame)
        instance._data = self._data.slice(start, length)
        instance._arrow_table = None
        instance.initialize()
        return instance

    # =========================================================================
    # Semantic Operations (inherited behavior from LineageMixin)
    # These methods build the data lineage graph for query optimization
    # =========================================================================
    
    def semantic_map(
        self,
        user_instruction: str,
        input_columns: list[str],
        output_columns: list[str],
        context: list[dict] | str | None = None,
        model: str | None = None,
        func: Callable | None = None,
        strategy: Literal["plain", "fewshot", "self-refine"] = "plain",
        limit: int | None = None,
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        """
        Apply a semantic map operation.
        
        Args:
            user_instruction: Natural language instruction for the LLM
            input_columns: Columns to use as input
            output_columns: New columns to generate
            context: Optional context for the LLM
            model: LLM model to use
            func: Optional custom function
            strategy: Execution strategy
            limit: Maximum number of rows to process
            rate_limit: Rate limit for API calls
            assertions: Optional list of assertion functions
        """
        op_kwargs = {
            "user_instruction": user_instruction,
            "input_columns": input_columns,
            "output_columns": output_columns,
            "context": context,
            "model": model,
            "tool": func,
            "strategy": strategy,
            "limit": limit,
            "rate_limit": rate_limit,
            "assertions": assertions,
        }
        data_kwargs = {
            "left_input_fields": self.leaf_node.node_fields.output_fields,
            "right_input_fields": [],
            "output_fields": self.leaf_node.node_fields.left_input_fields + output_columns,
        }
        self.add_operator(
            op_name="map",
            op_kwargs=op_kwargs,
            data_kwargs=data_kwargs,
            rate_limit=rate_limit
        )
        
    def semantic_filter(
        self,
        user_instruction: str,
        input_columns: list[str],
        func: Callable | None = None,
        context: list[dict] | str | None = None,
        model: str | None = None,
        strategy: Literal["plain", "fewshot", "self-refine"] = "plain",
        limit: int | None = None,
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        """
        Apply a semantic filter operation.
        
        Args:
            user_instruction: Natural language instruction for filtering
            input_columns: Columns to use as input
            func: Optional custom filter function
            context: Optional context for the LLM
            model: LLM model to use
            strategy: Execution strategy
            limit: Maximum number of rows to process
            rate_limit: Rate limit for API calls
            assertions: Optional list of assertion functions
        """
        op_kwargs = {
            "user_instruction": user_instruction,
            "input_columns": input_columns,
            "context": context,
            "model": model,
            "tool": func,
            "strategy": strategy,
            "limit": limit,
            "rate_limit": rate_limit,
            "assertions": assertions,
        }
        data_kwargs = {
            "left_input_fields": self.leaf_node.node_fields.output_fields,
            "right_input_fields": [],
            "output_fields": self.leaf_node.node_fields.output_fields,
        }
        self.add_operator(
            op_name="filter",
            op_kwargs=op_kwargs,
            data_kwargs=data_kwargs,
            rate_limit=rate_limit
        )
        
    def semantic_reduce(
        self,
        user_instruction: str,
        input_column: str,
        context: list[dict] | str | None = None,
        model: str | None = None,
        func: Callable | None = None,
        strategy: Literal["plain"] = "plain",
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        """
        Apply a semantic reduce/aggregation operation.
        
        Args:
            user_instruction: Natural language instruction for reduction
            input_column: Column to aggregate
            context: Optional context for the LLM
            model: LLM model to use
            func: Optional custom function
            strategy: Execution strategy
            rate_limit: Rate limit for API calls
            assertions: Optional list of assertion functions
        """
        op_kwargs = {
            "user_instruction": user_instruction,
            "input_columns": [input_column],
            "context": context,
            "model": model,
            "tool": func,
            "strategy": strategy,
            "rate_limit": rate_limit,
            "assertions": assertions,
        }
        data_kwargs = {
            "left_input_fields": self.leaf_node.node_fields.output_fields,
            "right_input_fields": [],
            "output_fields": []
        }
        self.add_operator(
            op_name="reduce",
            op_kwargs=op_kwargs,
            data_kwargs=data_kwargs,
            rate_limit=rate_limit
        )
        
    def semantic_join(
        self,
        other: "DataFrame",
        user_instruction: str,
        left_on: str,
        right_on: str,
        how: Literal["inner", "left", "right"] = "inner",
        context: list[dict] | str | None = None,
        model: str | None = None,
        func: Callable | None = None,
        strategy: Literal["nest", "block"] = "nest",
        limit: int | None = None,
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
        batch_size: int = 5,
    ):
        """
        Apply a semantic join operation.
        
        Args:
            other: The other DataFrame to join with
            user_instruction: Natural language instruction for join condition
            left_on: Column from this DataFrame to join on
            right_on: Column from other DataFrame to join on
            how: Join type (inner, left, right)
            context: Optional context for the LLM
            model: LLM model to use
            func: Optional custom function
            strategy: Join strategy (nest or block)
            limit: Maximum number of rows to process
            rate_limit: Rate limit for API calls
            assertions: Optional list of assertion functions
            batch_size: Batch size for processing
        """
        union_fields = (
            list(set(self.leaf_node.node_fields.output_fields) | 
                 set(other.leaf_node.node_fields.output_fields))
        )
        op_kwargs = {
            "user_instruction": user_instruction,
            "left_on": [left_on],
            "right_on": [right_on],
            "how": how,
            "context": context,
            "model": model,
            "tool": func,
            "strategy": strategy,
            "limit": limit,
            "rate_limit": rate_limit,
            "assertions": assertions,
            "batch_size": batch_size,
        }
        data_kwargs = {
            "input_left_fields": self.leaf_node.node_fields.output_fields,
            "input_right_fields": other.leaf_node.node_fields.output_fields,
            "output_fields": union_fields
        }
        self.add_operator(
            op_name="join",
            op_kwargs=op_kwargs,
            data_kwargs=data_kwargs,
            other=other,
            rate_limit=rate_limit
        )
        
    def semantic_rank(
        self,
        user_instruction: str,
        input_column: str,
        descend: bool = True,
        context: list[dict] | str | None = None,
        model: str | None = None,
        func: Callable | None = None,
        strategy: Literal["plain"] = "plain",
        limit: int | None = None,
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        """
        Apply a semantic ranking operation.
        
        Args:
            user_instruction: Natural language instruction for ranking
            input_column: Column to rank by
            descend: Sort descending (default: True)
            context: Optional context for the LLM
            model: LLM model to use
            func: Optional custom function
            strategy: Execution strategy
            limit: Maximum number of rows
            rate_limit: Rate limit for API calls
            assertions: Optional list of assertion functions
        """
        op_kwargs = {
            "user_instruction": user_instruction,
            "input_columns": [input_column],
            "descend": descend,
            "context": context,
            "model": model,
            "tool": func,
            "strategy": strategy,
            "limit": limit,
            "rate_limit": rate_limit,
            "assertions": assertions,
        }
        data_kwargs = {
            "left_input_fields": self.leaf_node.node_fields.output_fields,
            "right_input_fields": [],
            "output_fields": self.leaf_node.node_fields.output_fields,
        }
        self.add_operator(
            op_name="rank",
            op_kwargs=op_kwargs,
            data_kwargs=data_kwargs,
            rate_limit=rate_limit
        )
    
    def optimize_and_execute(self, optim_config=None):
        """
        Optimize the query plan and execute it.
        
        Args:
            optim_config: Optional optimization configuration
            
        Returns:
            Tuple of (output, cost, runtime)
        """
        self.create_plan_optimizer(optim_config)
        if self.optimizer.config.do_logical_optimization:
            self.leaf_node = self.optimizer.optimize_logical_plan(self.leaf_node)
        if self.optimizer.config.do_physical_optimization:
            output, cost, runtime = self.optimizer.optimize_physical_plan(
                self.leaf_node,
            )
        else:
            output, cost, runtime = self.execute()
        return output, cost, runtime
