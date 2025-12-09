# Rank Operation

The Rank operation ranks records in a column of input data based on a natural language condition.

## nirvana.ops.rank

`(input_data: pandas.DataFrame, user_instruction: str, input_column: str, descend: bool = True, func: Callable = None, **kwargs) -> RankOpOutputs`

A function wrapper that invokes `RankOperation.execute(input_data)`

## RankOpOutputs

The type of return value of `RankOperation` (base: `BaseOpOutputs`).

**Parameters:**
- `ranked_indices` (`list[int]`): List of indices of sorted records
- `ranking` (`list[int]`): Ranking of sorted records
- `cost` (`float`): Token cost for executing map operation  

## RankOperation

Core class to implement rank operation (base: `BaseOperation`).

**Parameters:**

| **Name** | **Type** | **Description** | **Default** |
| --- | --- | --- | --- |
| `user_instruction` | `str` | User instruction | required |
| `input_columns` | `list[str]` | Input columns that the rank operation evaluates on | required |
| `descend` | `bool` | Whether to sort in descending order (i.e., from best satisfied to least satisfied) | `True` |
| `context` | `dict` | Additional context information (e.g., demonstrations) for LLM reasoning | `{}` |
| `model` | `str` | LLM backend for operation execution | `None` |
| `tool` | `BaseTool` | Function tools for operation execution | `None` |
| `semaphore` | `int` | Concurrent LLM calls | `16` |
| `assertions` | `list[Callable]` | Assertions/guardrails to constrain behavior of map operation | `[]` |
| `strategy` | `Literal["plain"]` | The rank algorithm | `"plain"` |

### Properties:
- `dependencies` (`list(str)`): Dependent columns of the rank operation
- `generated_fields` (`list(str)`): Names of generated fields of the operation
- `op_kwargs` (`dict`): arguments in the rank operation

### Methods

#### `execute(input_data, *args, **kwargs) -> RankOpOutputs`
