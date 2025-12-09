# nirvana.DataFrame

## `class DataFrame(LineageMixin)`

A DataFrame class that extends pandas.DataFrame with semantic operations and data lineage tracking that enables lazy execution and query optimization.

### Constructor

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

### Properties

- `columns`: List of column names
- `nrows`: Number of rows
- `primary_key`: Primary key column (if set)

### Class Methods

#### `from_external_file(path: str, sep=',', **kwargs)`
Load a DataFrame from an external file.

**arguments:**

- `path` (str): Path to the file
- `sep` (str): (default: ',') Delimiter
- `**kwargs`: Additional arguments passed to `pd.read_table()`

**Returns:**

- `DataFrame`: A new DataFrame instance

### Methods

#### `semantic_map(user_instruction: str, input_column: str, output_column: str, rate_limit: int = 16)`

Add a map operation to the data lineage alongwith the dataframe.

**arguments:**

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

#### `semantic_filter(user_instruction: str, input_column: str, rate_limit: int = 16)`

Add a filter operation to the data lineage alongwith the dataframe.

**arguments:**

- `user_instruction` (str): Natural language filtering condition
- `input_column` (str): Column to evaluate
- `rate_limit` (int): Maximum concurrent LLM calls (default: 16)

**Example:**
```python
df.semantic_filter(
    user_instruction="Whether the comment is positive",
    input_column="Comment"
)
```

#### `semantic_join(other: nirvana.DataFrame, user_instruction: str, left_on: str, right_on: str, how: str, rate_limit: int = 16)`

Add a join operation to the data lineage alongwith the dataframe.

**arguments:**

- `other` (nirvana.DataFrame): Right DataFrame to join
- `user_instruction` (str): Natural language join condition
- `left_on` (str): Column from left DataFrame
- `right_on` (str): Column from right DataFrame
- `how` (str): Join type: "inner", "left", or "right"
- `rate_limit` (int): Maximum concurrent LLM calls (default: 16)

**Example:**
```python
symptom.semantic_join(
    other=drug,
    user_instruction="Does the drug cure the symptom?",
    left_on="symptom",
    right_on="medical_use",
    how="inner"
)
```

#### `semantic_rank(user_instruction: str, input_column: str, descend: bool = True, rate_limit: int = 16)`

Add a rank operation to the data lineage alongwith the dataframe.

**arguments:**

- `user_instruction` (str): Natural language join condition
- `input_column` (str): Column to evaluate
- `descend` (bool): if `descend=True`, sort records from the most-satisfied to the least-satisfied; otherwise, sort them from the least-satisfied to the most-satisfied (default: `True`)
- `rate_limit` (int): Maximum concurrent LLM calls (default: 16)

**Example:**
```python
course.semantic_rank(
    user_instruction="The course requires the least math",
    input_column="Courses",
    descend=True,
)
```

#### `semantic_reduce(user_instruction: str, input_column: str, rate_limit: int = 16)`

Add an aggregation operation to the data lineage alongwith the dataframe.

**arguments:**

- `user_instruction` (str): Natural language aggregation instruction
- `input_column` (str): Column to aggregate
- `rate_limit` (int): Maximum concurrent LLM calls (default: 16)

**Example:**
```python
df.semantic_reduce(
    user_instruction="Summarize the plot structure",
    input_column="Plot"
)
```

#### `execute() -> tuple[pd.DataFrame, float, float]`

Execute the query plan along the data lineage.

**Returns:**

- `tuple`: (output DataFrame, total cost, runtime in seconds)

#### `optimize_and_execute(optim_config: OptimizeConfig = None) -> tuple[pd.DataFrame, float, float]`

Optimize and execute the query plan.

**Arguments:**

- `optim_config` (nirvana.OptimizeConfig): configurations for optimization

**Returns:**

- `tuple`: (result DataFrame, total cost, runtime in seconds)

#### `print_lineage_graph(op_signature_width: int = 100, max_instruction_print_length: int = 50)`

Print a visual representation of the query.

**Arguments:**

- `op_signature_width` (int): Width for operator signatures
- `max_instruction_print_length` (int): Max length for instruction display

#### `clear_lineage_graph()`

Clear the data lineage graph alongwith the dataframe.
