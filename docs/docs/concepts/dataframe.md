# DataFrame

The `DataFrame` class is the central user-facing abstraction in Nirvana. It combines the familiarity of pandas DataFrames with powerful semantic operations and lineage tracking.

## Overview

A Nirvana `DataFrame` wraps a pandas DataFrame and provides a fluent API for lazy semantic operations. It inherits from `LineageMixin` to automatically track operations and build a lineage graph.

```python
from nirvana.dataframe.frame import DataFrame

df = DataFrame(data)
```

## Key Features

### 1. Hybrid Data Support
Nirvana DataFrames handle standard tabular data as well as unstructured data like text, images, and audio.

### 2. Semantic Operations
Unlike standard pandas operations which execute immediately, Nirvana's semantic operations are **lazy** and **declarative**. They describe *what* to do, not *how* to do it. The actual execution happens only when `optimize_and_execute()` is called (or implicitly triggered).

- `semantic_map()`: Transform data using LLMs (e.g., sentiment analysis, extraction).
- `semantic_filter()`: Filter rows based on natural language criteria.
- `semantic_join()`: Join two DataFrames fuzzy or semantically.
- `semantic_rank()`: Rank rows based on relevance or quality.
- `semantic_reduce()`: Aggregate data (e.g., summarization).

```python
# Example:
df.semantic_map(
    user_instruction="Extract the sentiment",
    input_columns=["review"],
    output_columns=["sentiment"]
)
```

### 3. Lineage Tracking
As you chain operations, `LineageMixin` builds a DAG of operators. This allows Nirvana to optimize the entire pipeline before execution (details in [Data Lineage](data_lineage.md)).

```python
# Stacks operators in the lineage graph
df = df.semantic_filter(...) \
       .semantic_map(...) \
       .semantic_rank(...)
```

### 4. Optimization & Execution
The `optimize_and_execute` method triggers the pipeline:

1.  **Logical Optimization**: Rewrites the plan (e.g., filter pushdown, operator fusion) to reduce cost and improve performance.
2.  **Physical Optimization**: Selects the best LLM models or execution strategies for each operator.
3.  **Execution**: Runs the optimized plan asynchronously.

```python
result_df, cost, time = df.optimize_and_execute()
```

## Interoperability
You can convert a Nirvana DataFrame back to a pandas DataFrame using `to_pandas()` or access the underlying data via `_data`.
