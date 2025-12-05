# API Reference

Complete API reference for Nirvana's core components.

## DataFrame

### `nirvana.DataFrame`

A DataFrame class that extends pandas DataFrame with semantic operations and data lineage tracking.

#### Constructor

```python
DataFrame(data: pd.DataFrame = None, *args, **kwargs)
```

**Parameters:**
- `data` (pd.DataFrame): A pandas DataFrame to wrap

**Example:**
```python
import pandas as pd
import nirvana as nv

df = nv.DataFrame(pd.DataFrame({"col1": [1, 2, 3]}))
```

#### Class Methods

##### `from_external_file(path: str, sep=',', **kwargs) -> DataFrame`

Load a DataFrame from an external file.

**Parameters:**
- `path` (str): Path to the file
- `sep` (str): Delimiter (default: ',')
- `**kwargs`: Additional arguments passed to `pd.read_table()`

**Returns:**
- `DataFrame`: A new DataFrame instance

#### Properties

- `columns`: List of column names
- `nrows`: Number of rows
- `primary_key`: Primary key column (if set)

#### Semantic Operations

##### `semantic_map(user_instruction: str, input_column: str, output_column: str, rate_limit: int = 16)`

Perform element-wise transformation to create a new column.

**Parameters:**
- `user_instruction` (str): Natural language instruction for the transformation
- `input_column` (str): Column to transform
- `output_column` (str): Name of the new column
- `rate_limit` (int): Maximum concurrent LLM calls (default: 16)

**Example:**
```python
df.semantic_map(
    user_instruction="Extract the genre from the overview",
    input_column="overview",
    output_column="genre"
)
```

##### `semantic_filter(user_instruction: str, input_column: str, rate_limit: int = 16)`

Filter rows based on a natural language condition.

**Parameters:**
- `user_instruction` (str): Natural language condition
- `input_column` (str): Column to evaluate
- `rate_limit` (int): Maximum concurrent LLM calls (default: 16)

**Example:**
```python
df.semantic_filter(
    user_instruction="Rating is higher than 8.5",
    input_column="IMDB_Rating"
)
```

##### `semantic_reduce(user_instruction: str, input_column: str, rate_limit: int = 16)`

Aggregate values in a column into a single result.

**Parameters:**
- `user_instruction` (str): Natural language aggregation instruction
- `input_column` (str): Column to aggregate
- `rate_limit` (int): Maximum concurrent LLM calls (default: 16)

**Example:**
```python
df.semantic_reduce(
    user_instruction="Count the number of crime movies",
    input_column="genre"
)
```

##### `semantic_join(other: DataFrame, user_instruction: str, left_on: str, right_on: str, how: str, rate_limit: int = 16)`

Join two DataFrames based on semantic similarity.

**Parameters:**
- `other` (DataFrame): Right DataFrame to join
- `user_instruction` (str): Natural language join condition
- `left_on` (str): Column from left DataFrame
- `right_on` (str): Column from right DataFrame
- `how` (str): Join type: "inner", "left", or "right"
- `rate_limit` (int): Maximum concurrent LLM calls (default: 16)

**Example:**
```python
df1.semantic_join(
    other=df2,
    user_instruction="Do these match semantically?",
    left_on="symptom",
    right_on="medical_use",
    how="inner"
)
```

#### Execution Methods

##### `execute() -> tuple[pd.DataFrame, float, float]`

Execute the query plan along the data lineage.

**Returns:**
- `tuple`: (output DataFrame, total cost, runtime in seconds)

##### `optimize_and_execute(optim_config: OptimizeConfig = None) -> tuple[pd.DataFrame, float, float]`

Optimize and execute the query plan.

**Parameters:**
- `optim_config` (OptimizeConfig): Optimization configuration

**Returns:**
- `tuple`: (output DataFrame, total cost, runtime in seconds)

#### Utility Methods

##### `print_lineage_graph(op_signature_width: int = 100, max_instruction_print_length: int = 50)`

Print a visual representation of the data lineage graph.

**Parameters:**
- `op_signature_width` (int): Width for operator signatures
- `max_instruction_print_length` (int): Max length for instruction display

##### `clear_lineage_graph()`

Clear the data lineage graph.

## Operators

### Direct Operator Usage

You can also use operators directly without DataFrame:

#### `nirvana.ops.map`

```python
from nirvana.ops import map

outputs = map.map_wrapper(
    input_data=df,
    user_instruction="Extract genre",
    input_column="overview",
    output_columns=["genre"],
    strategy="plain"
)
```

**Returns:** `MapOpOutputs` with:
- `field_name`: List of output column names
- `output`: Dictionary mapping column names to values
- `cost`: Total token cost

#### `nirvana.ops.filter`

```python
from nirvana.ops import filter

outputs = filter.filter_wrapper(
    input_data=df,
    user_instruction="Rating > 8.5",
    input_column="IMDB_Rating",
    strategy="plain"
)
```

**Returns:** `FilterOpOutputs` with:
- `output`: List of boolean values
- `cost`: Total token cost

#### `nirvana.ops.reduce`

```python
from nirvana.ops import reduce

outputs = reduce.reduce_wrapper(
    input_data=df,
    user_instruction="Count crime movies",
    input_column="genre"
)
```

**Returns:** `ReduceOpOutputs` with:
- `output`: Aggregated result
- `cost`: Total token cost

#### `nirvana.ops.join`

```python
from nirvana.ops import join

outputs = join.join_wrapper(
    left_data=df1,
    right_data=df2,
    user_instruction="Do these match?",
    left_on="col1",
    right_on="col2",
    how="inner"
)
```

**Returns:** `JoinOpOutputs` with:
- `joined_pairs`: List of (left_idx, right_idx) tuples
- `left_join_keys`: List of left join keys
- `right_join_keys`: List of right join keys
- `cost`: Total token cost

## Query Optimization

### `nirvana.optim.OptimizeConfig`

Configuration class for query optimization.

#### Constructor

```python
OptimizeConfig(
    do_logical_optimization: bool = True,
    do_physical_optimization: bool = True,
    sample_ratio: Optional[float] = None,
    sample_size: Optional[int] = None,
    improve_margin: float = 0.2,
    approx_mode: bool = True,
    filter_pullup: bool = True,
    filter_pushdown: bool = True,
    map_pullup: bool = True,
    non_llm_pushdown: bool = True,
    non_llm_replace: bool = True,
    avaiable_models: list[str] = []
)
```

**Parameters:**

**Optimization Flags:**
- `do_logical_optimization` (bool): Enable logical plan optimization (default: True)
- `do_physical_optimization` (bool): Enable physical plan optimization (default: True)

**Physical Optimization:**
- `sample_ratio` (float, optional): Ratio of data for optimization (0.0-1.0)
- `sample_size` (int, optional): Number of samples for optimization
- `improve_margin` (float): Minimum improvement threshold (default: 0.2)
- `approx_mode` (bool): Use approximation mode (default: True)
- `avaiable_models` (list[str]): Available models for selection

**Logical Optimization Rules:**
- `filter_pullup` (bool): Enable filter pullup (default: True)
- `filter_pushdown` (bool): Enable filter pushdown (default: True)
- `map_pullup` (bool): Enable map pullup (default: True)
- `non_llm_pushdown` (bool): Enable non-LLM pushdown (default: True)
- `non_llm_replace` (bool): Enable non-LLM replacement (default: True)

**Example:**
```python
config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    do_physical_optimization=True,
    sample_size=10,
    improve_margin=0.15,
    non_llm_replace=False  # Disable specific rule
)
```

### `nirvana.optim.PlanOptimizer`

Plan optimizer class (typically used internally).

#### Methods

##### `optimize_logical_plan(plan: LineageNode) -> LineageNode`

Optimize the logical plan.

**Parameters:**
- `plan` (LineageNode): Root node of the lineage graph

**Returns:**
- `LineageNode`: Optimized plan

##### `optimize_physical_plan(plan: LineageNode, num_records: int) -> tuple`

Optimize the physical plan.

**Parameters:**
- `plan` (LineageNode): Root node of the lineage graph
- `num_records` (int): Number of records in the dataset

**Returns:**
- `tuple`: (output DataFrame, cost, runtime)

## Configuration

### `nirvana.configure_llm_backbone`

Configure the LLM backend for all operations.

```python
nirvana.configure_llm_backbone(
    model_name: str = None,
    api_key: Union[str, Path] = None,
    base_url: str = None,
    **kwargs
)
```

**Parameters:**
- `model_name` (str): Model name (e.g., "gpt-4o", "deepseek-chat")
- `api_key` (str | Path): API key or path to API key file
- `base_url` (str, optional): Base URL (inferred from model_name if not provided)
- `**kwargs`: Additional arguments for LLMClient

**Example:**
```python
import nirvana as nv

nv.configure_llm_backbone(
    model_name="gpt-4o",
    api_key="sk-...",
    max_tokens=512,
    temperature=0.1
)
```

## Data Types

### `nirvana.ImageArray`

Array type for image data.

```python
from nirvana import ImageArray

images = ImageArray([
    "https://example.com/image1.png",
    "/path/to/image2.jpg"
])
```

### `nirvana.ImageDtype`

Pandas extension dtype for images.

```python
from nirvana import ImageDtype

# Automatically registered with pandas
df = pd.DataFrame({
    "images": ImageArray([...])
})
```

## Internal Classes

### `nirvana.lineage.LineageNode`

Represents a node in the data lineage graph.

**Properties:**
- `op_name`: Operator name
- `operator`: Operator instance
- `node_fields`: Field metadata
- `left_child`: Left child node
- `right_child`: Right child node

**Methods:**
- `execute_operation(input)`: Execute the operation
- `run(input)`: Run and return NodeOutput

### `nirvana.ops.base.BaseOperation`

Base class for all operators.

**Properties:**
- `op_name`: Operator name
- `user_instruction`: Natural language instruction
- `model`: LLM model to use
- `implementation`: Implementation strategy

**Methods:**
- `execute(input_data, **kwargs)`: Abstract method to execute the operation

