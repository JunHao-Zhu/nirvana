# Tutorial

This tutorial covers the core concepts and features of Nirvana, from basic operations to advanced query optimization.

## DataFrame

Nirvana's `DataFrame` is built on pandas DataFrame and adds support for unstructured data types and semantic operations.

### Creating DataFrames

```python
import pandas as pd
import nirvana as nv

# From pandas DataFrame
df = nv.DataFrame(pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}))

# From external file
df = nv.DataFrame.from_external_file("data.csv", sep=",")

# With image data
logo_imgs = nv.ImageArray([
    "https://spark.apache.org/images/spark-logo.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/488px-PyTorch_logo_black.svg.png"
])
df = nv.DataFrame(pd.DataFrame({
    "names": ["Spark", "Pytorch"],
    "logos": logo_imgs
}))
```

### DataFrame Properties

```python
# Access columns
print(df.columns)  # ['col1', 'col2']

# Get number of rows
print(len(df))  # 3

# Check column membership
print("col1" in df)  # True
```

## Operators

Nirvana provides semantic operators that use LLMs to process data based on natural language instructions.

### Map Operator

The `map` operator performs element-wise transformations, creating new columns based on natural language instructions.

```python
df = nv.DataFrame(pd.DataFrame({
    "title": ["The Godfather", "The Dark Knight"],
    "overview": [
        "An organized crime dynasty's aging patriarch transfers control...",
        "When the menace known as the Joker wreaks havoc..."
    ]
}))

# Extract genre from overview
df.semantic_map(
    user_instruction="According to the movie overview, extract the genre of each movie.",
    input_column="overview",
    output_column="genre"
)
```

**Implementation Strategies:**
- `plain`: Direct LLM call (default)
- `self-refine`: Generate, evaluate, and refine if needed
- `fewshot`: Use in-context learning with examples

### Filter Operator

The `filter` operator evaluates conditions on each row, returning boolean values.

```python
# Filter movies released after 2000
df.semantic_filter(
    user_instruction="Whether the movie is released after 2000?",
    input_column="title"
)

# Filter by rating
df.semantic_filter(
    user_instruction="The rating is higher than 8.5.",
    input_column="IMDB_Rating"
)
```

### Reduce Operator

The `reduce` operator aggregates values in a column into a single result.

```python
# Find common themes
df.semantic_reduce(
    user_instruction="Based on the overviews, provide several common points of these movies.",
    input_column="overview"
)

# Count specific items
df.semantic_reduce(
    user_instruction="Count the number of crime movies.",
    input_column="genre"
)
```

**Note:** The current implementation is simple. Future versions will support optimizations like "summarize and aggregate" and "incremental aggregation."

### Join Operator

The `join` operator joins two DataFrames based on semantic similarity between columns.

```python
# Clinical notes
clinical_note = nv.DataFrame(pd.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [12, 20],
    "gender": ["F", "M"],
    "symptom": ["headache", "have a cough"]
}))

# Drug database
drug = nv.DataFrame(pd.DataFrame({
    "name": ["Salbutamol", "ibuprofen"],
    "medical_use": [
        "treat bronchospasm, as well as chronic obstructive pulmonary disease",
        "treat mild to moderate pain, painful menstruation, osteoarthritis, dental pain, headaches, and pain from kidney stones"
    ]
}))

# Semantic join
clinical_note.semantic_join(
    other=drug,
    user_instruction="Does the drug cure the possible disease according to the symptoms?",
    left_on="symptom",
    right_on="medical_use",
    how="inner"  # or "left", "right"
)
```

**Join Types:**
- `inner`: Keep only matching pairs
- `left`: Keep all left rows, match with right where possible
- `right`: Keep all right rows, match with left where possible

**Note:** The current implementation has quadratic complexity. Optimizations are planned.

### Rank Operator

The `rank` operator ranks rows based on a natural language criterion.

```python
df.semantic_rank(
    user_instruction="rank the movies by their relevance to DC Comics.",
    input_column="title"
)
```

## Plan Optimization

Nirvana supports two levels of query optimization: logical and physical plan optimization.

### Logical Plan Optimization

Logical optimization applies rule-based transformations to improve query efficiency:

```python
df = nv.DataFrame(pd.read_csv("movie_data.csv"))

# Build a query
df.semantic_map(
    user_instruction="According to the movie overview, extract the genre of each movie.",
    input_column="Overview",
    output_column="Genre"
)
df.semantic_filter(
    user_instruction="The rating is higher than 7.",
    input_column="IMDB_Rating"
)
df.semantic_filter(
    user_instruction="The rating is lower than 8.",
    input_column="IMDB_Rating"
)
df.semantic_reduce(
    user_instruction="Count the number of crime movies.",
    input_column="Genre"
)

# View the logical plan
df.print_lineage_graph()

# Optimize and execute
config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    do_physical_optimization=False
)
output, cost, runtime = df.optimize_and_execute(optim_config=config)
```

**Transformation Rules:**

1. **Non-LLM Replacement**: Replaces NL instructions over non-image/video/audio data with equivalent compute functions when possible.

2. **Map Pullup**: Moves map operations to the top of the query plan when beneficial.

3. **Filter Pullup**: Identifies cases where a filter can be applied on columns in other tables.

4. **Filter Pushdown**: Pushes filters down into the query plan and duplicates filters over equivalency sets.

5. **Non-LLM Pushdown**: Pushes operators using non-LLM functions/tools down into the query plan.

**Configuring Rules:**

```python
config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    non_llm_replace=True,      # Enable/disable specific rules
    filter_pullup=True,
    filter_pushdown=True,
    map_pullup=True,
    non_llm_pushdown=True
)
```

### Physical Plan Optimization

Physical optimization selects the most cost-effective LLM model for each operator:

```python
config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    do_physical_optimization=True,
    sample_size=5,              # Number of samples for optimization
    improve_margin=0.2,         # Minimum improvement threshold
    approx_mode=True,           # Use approximation mode
    avaiable_models=["gpt-4o", "gpt-3.5-turbo", "deepseek-chat"]
)

output, cost, runtime = df.optimize_and_execute(optim_config=config)
```

**Parameters:**
- `sample_size`: Number of data samples to use for model selection
- `sample_ratio`: Alternative to `sample_size`, specifies ratio of data
- `improve_margin`: Minimum cost improvement (0.0-1.0) to switch models
- `approx_mode`: Whether to use approximation for faster optimization

## Data Lineage

Every DataFrame operation builds a data lineage graph (DAG) that tracks:
- Operator sequence
- Input/output fields
- Dependencies between operations

```python
# Build a query
df.semantic_map(...)
df.semantic_filter(...)

# View the lineage graph
df.print_lineage_graph()

# Execute along the lineage
output, cost, runtime = df.execute()
```

The lineage graph enables:
- **Lazy Execution**: Operations are not executed until `execute()` or `optimize_and_execute()` is called
- **Plan Optimization**: The optimizer can transform the graph
- **Cost Tracking**: Track token costs for each operation

## Best Practices

1. **Batch Operations**: Use `rate_limit` to control concurrent LLM calls:
   ```python
   df.semantic_map(..., rate_limit=16)  # Max 16 concurrent calls
   ```

2. **Use UDFs When Possible**: For simple operations, provide Python functions instead of natural language:
   ```python
   df.semantic_filter(
       user_instruction="Rating > 8.5",
       input_column="rating",
       func=lambda x: x > 8.5
   )
   ```

3. **Optimize Selectively**: Turn off optimization for simple queries:
   ```python
   config = nv.optim.OptimizeConfig(
       do_logical_optimization=False,
       do_physical_optimization=False
   )
   ```

4. **Monitor Costs**: Check the cost returned by `optimize_and_execute()`:
   ```python
   output, cost, runtime = df.optimize_and_execute()
   print(f"Total cost: ${cost:.4f}")
   ```

