# Reduce Operation

The Reduce operation aggregates multiple values in a given column into a single result according to the user's instruction.

## nirvana.ops.reduce

`(input_data: pandas.DataFrame, user_instruction: str, input_column: str, func: Callable = None, strategy: str = "plain", **kwargs) -> ReduceOpOutputs`

A function wrapper that invokes `ReduceOperation.execute(input_data)`

## ReduceOpOutputs

The type of return value of `ReduceOperation` (base: `BaseOpOutputs`).

**Parameters:**
- `output` (`Any`): The aggregation results
- `cost` (`float`): Token cost for executing map operation  

## ReduceOperation

Core class to implement reduce operation (base: `BaseOperation`).

**Parameters:**

| **Name** | **Type** | **Description** | **Default** |
| --- | --- | --- | --- |
| `user_instruction` | `str` | User instruction | required |
| `input_columns` | `list[str]` | Input columns that the rank operation evaluates on | required |
| `context` | `dict` | Additional context information (e.g., demonstrations) for LLM reasoning | `{}` |
| `model` | `str` | LLM backend for operation execution | `None` |
| `tool` | `BaseTool` | Function tools for operation execution | `None` |
| `semaphore` | `int` | Concurrent LLM calls | `16` |
| `assertions` | `list[Callable]` | Assertions/guardrails to constrain behavior of map operation | `[]` |
| `strategy` | `Literal["plain"]` | LLM inference strategy | `"plain"` |

### Properties:
- `dependencies` (`list(str)`): Dependent columns of the reduce operation
- `generated_fields` (`list(str)`): Names of generated fields of the reduce operation
- `op_kwargs` (`dict`): arguments in the reduce operation

### Methods

#### `execute(input_data, *args, **kwargs) -> ReduceOpOutputs`
