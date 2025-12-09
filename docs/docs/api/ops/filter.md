# Filter Operation

The Filter operator evaluates an NL condition on each item in a column.

## nirvana.ops.filter

`(input_data: pandas.DataFrame, user_instruction: str, input_column: str, func: Callable = None, strategy: Literal["plain", "fewshot", "self-refine"] = "plain", **kwargs) -> FilterOpOutputs`

A function wrapper that invokes `FilterOperation.execute(input_data)`

## FilterOpOutputs

The type of return value of `FilterOperation` (base: `BaseOpOutputs`).

**Parameters:**
- `output` (`Iterable[bool]`): List of evaluation results
- `cost` (`float`): Token cost for executing map operation  

## FilterOperation

Core class to implement filter operation (base: `BaseOperation`).

**Parameters:**

| **Name** | **Type** | **Description** | **Default** |
| --- | --- | --- | --- |
| `user_instruction` | `str` | User instruction | required |
| `input_columns` | `list[str]` | List of input columns | required |
| `context` | `dict` | Additional context information (e.g., demonstrations) for LLM reasoning | `{}` |
| `model` | `str` | LLM backend for operation execution | `None` |
| `tool` | `BaseTool` | Function tools for operation execution | `None` |
| `semaphore` | `int` | Concurrent LLM calls | `16` |
| `assertions` | `list[Callable]` | Assertions/guardrails to constrain behavior of map operation | `[]` |
| `strategy` | `Literal["plain", "fewshot", "self-refine"]` | The workflow of LLM inference: `plain` represents a direct LLM inference; `fewshot` represents LLM inference with few-shot demos (requiring demos in `context`); `self-refine` represents self-refinement workflow that gives feedback to the initial generation then refines LLM outputs accordingly | `"plain"` |

### Properties:
- `dependencies` (`list(str)`): Dependent columns of the filter operation
- `generated_fields` (`list(str)`): Names of generated fields of the filter operation
- `op_kwargs` (`dict`): arguments in the filter operation

### Methods

#### `execute(input_data, *args, **kwargs) -> FilterOpOutputs`
