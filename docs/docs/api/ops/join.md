# Join Operation

The Join operation joins two tables by keeping tuple pairs that satisfy a natural language condition.

## nirvana.ops.join

`(left_data: pandas.DataFrame, right_data: pandas.DataFrame, user_instruction: str, left_on: str, right_on: str, how: str = "inner", strategy: Literal["nest", "block"] = "nest", **kwargs) -> JoinOpOutputs`

A function wrapper that invokes `JoinOperation.execute(left_data, right_data)`

## JoinOpOutputs

The type of return value of `JoinOperation` (base: `BaseOpOutputs`).

**Parameters:**
- `joined_pairs` (`list[tuple]`): List of indices pairs of joined records
- `left_join_keys` (`list[int]`): Join keys for the left table
- `right_join_keys` (`list[int]`): Join keys for the right table
- `cost` (`float`): Token cost for executing map operation

## JoinOperation

Core class to implement join operation (base: `BaseOperation`).

**Parameters:**

| **Name** | **Type** | **Description** | **Default** |
| --- | --- | --- | --- |
| `user_instruction` | `str` | User instruction | required |
| `left_on` | `list[str]` | Input columns from the left table that the operation evaluates on | required |
| `right_on` | `list[str]` | Input columns from the right table that the operation evaluates on | required |
| `how` | `str` | Join type (`inner`, `left`, `right`) | `"inner"` |
| `context` | `dict` | Additional context information (e.g., demonstrations) for LLM reasoning | `{}` |
| `model` | `str` | LLM backend for operation execution | `None` |
| `tool` | `BaseTool` | Function tools for operation execution | `None` |
| `semaphore` | `int` | Concurrent LLM calls | `16` |
| `assertions` | `list[Callable]` | Assertions/guardrails to constrain behavior of map operation | `[]` |
| `strategy` | `Literal["nest", "block"]` | The join algorithm adopted the operation: `nest` represents pair-wise comparisons for nested-loop join; `block` represents block-wise joined pair identification | `"nest"` |

### Properties:
- `dependencies` (`list(str)`): Dependent columns of the join operation
- `generated_fields` (`list(str)`): Names of generated fields of the join operation
- `op_kwargs` (`dict`): arguments in the join operation

### Methods

#### `execute(input_data, *args, **kwargs) -> JoinOpOutputs`
