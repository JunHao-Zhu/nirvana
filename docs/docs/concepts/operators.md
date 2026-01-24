# Operators

Nirvana provides a set of semantic operators that allow you to manipulate data using natural language instructions. These operators form the nodes of the data lineage graph.

## Supported Operators

### Scan
The **Scan** operator is the entry point of the lineage. It represents the initial loading of data into the system.
- **Input**: A data source (e.g., in-memory DataFrame, external file).
- **Output**: A DataFrame containing the loaded data.

### Map (`semantic_map`)
The **Map** operator performs row-wise transformations. It applies a user instruction to each row to generate new columns.
- **Use Cases**: Sentiment analysis, information extraction, translation, style transfer.
- **Input**: Specific columns from the DataFrame.
- **Output**: New columns appended to the DataFrame.

### Filter (`semantic_filter`)
The **Filter** operator selects rows based on a boolean condition derived from the user instruction.
- **Use Cases**: Removing irrelevant data, selecting specific items.
- **Input**: Specific columns to evaluate.
- **Output**: A subset of the original rows.

### Rank (`semantic_rank`)
The **Rank** operator reorders rows based on their relevance or a specified quality metric.
- **Use Cases**: Top-k retrieval, prioritizing importance.
- **Input**: Input column to rank by.
- **Output**: A reordered DataFrame (often used with a limit).

### Reduce (`semantic_reduce`)
The **Reduce** operator aggregates the data into a single result or a summary.
- **Use Cases**: Summarization, key insight extraction.
- **Input**: Input column to aggregate.
- **Output**: A single-row DataFrame containing the result.

### Join (`semantic_join`)
The **Join** operator combines two DataFrames based on a semantic relationship or fuzzy matching key.
- **Use Cases**: Merging datasets with different schemas, linking entities.
- **Input**: Left and Right DataFrames, join keys/conditions.
- **Output**: A merged DataFrame.

## Operator Interface

All operators inherit from `BaseOperation` and implement the `execute` method:

```python
class BaseOperation:
    async def execute(self, input_data: Any, **kwargs) -> BaseOpOutputs:
        pass
```

The execution is typically handled by an LLM (Semantic Operator) or a deterministic function (if provided).
