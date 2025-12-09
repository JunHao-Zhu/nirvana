# Map Operation

The Map operation applies a specified transformation in natural language to each item in a column of input data

## nirvana.ops.map

`( input_data: pandas.DataFrame, user_instruction: str, input_column: str, output_columns: list[str], func: Callable = None, strategy: Literal["plain", "fewshot", "self-refine"] = "plain", **kwargs ) -> MapOpOutputs`

A function wrapper that invokes `MapOperation.execute(input_data)`

## MapOpOutputs

The type of return value of `MapOperation` (base: `BaseOpOutputs`).

**Parameters:**

- `field_name` (`list[str]`): List of output column names
- `output` (`dict[str, list]`): Output values for the generated fields
- `cost` (`float`): Token cost for executing map operation  

## MapOperation

Core class to implement map operation (base: `BaseOperation`).

**Parameters:**


### Parameters

| **Name** | **Type** | **Description** | **Default** |
| --- | --- | --- | --- |
| `user_instruction` | `str` | User instruction | required |
| `input_columns` | `list[str]` | List of input columns | required |
| `output_columns` | `list[str]` | List of names of generated fields | required |
| `context` | `dict` | Additional context information (e.g., demonstrations) for LLM reasoning | `{}` |
| `model` | `str` | LLM backend for operation execution | `None` |
| `tool` | `BaseTool` | Function tools for operation execution | `None` |
| `semaphore` | `int` | Concurrent LLM calls | `16` |
| `assertions` | `list[Callable]` | Assertions/guardrails to constrain behavior of map operation | `[]` |
| `strategy` | `Literal["plain", "fewshot", "self-refine"]` | The workflow of LLM inference: `plain` represents a direct LLM inference; `fewshot` represents LLM inference with few-shot demos (requiring demos in `context`); `self-refine` represents self-refinement workflow that gives feedback to the initial generation then refines LLM outputs accordingly | `"plain"` |

### Properties:

- `dependencies` (`list(str)`): Dependent columns of the map operation
- `generated_fields` (`list(str)`): Names of generated fields of the map operation
- `op_kwargs` (`dict`): arguments in the map operation

### Methods

#### `execute(input_data, *args, **kwargs) -> MapOpOutputs`
