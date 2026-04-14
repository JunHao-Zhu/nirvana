# Nirvana
<!--- BADGES: START --->
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2511.19830)
[![PyPI](https://img.shields.io/pypi/v/nirvana-ai)](https://pypi.org/project/nirvana-ai/)
[![Documentation](https://img.shields.io/badge/Documentation-docs-green)](https://JunHao-Zhu.github.io/nirvana)
<!--- BADGES: END --->

**Nirvana** is a multi-modal data analytics framework that brings LLM-driven semantic
operators to pandas DataFrames. It provides two complementary interfaces — an eager
API for interactive use, and a lazy DataFrame API that enables logical and physical
query plan optimization.

> 📖 **Documentation**: Full documentation is available at [docs/](docs/) or build
> locally with `mkdocs serve`.

If you find Nirvana useful, please consider citing our paper:

> Junhao Zhu, Lu Chen, Xiangyu Ke, Ziquan Fang, Tianyi Li, Yunjun Gao, Christian S.
> Jensen. *Beyond Relational: Semantic-Aware Multi-Modal Analytics with LLM-Native
> Query Optimization.* arXiv:2511.19830, 2025.

---

## Installation

Install from PyPI:

```bash
pip install nirvana-ai
```

Install the latest development version from source:

```bash
pip install git+https://github.com/JunHao-Zhu/nirvana.git
```

---

## Quick Start

```python
import os
import pandas as pd
import nirvana as nv

nv.configure_llm_backbone(
    model_name="gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

df = pd.DataFrame({
    "title": ["The Godfather", "The Dark Knight", "Inception"],
    "overview": [
        "An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.",
        "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
        "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.",
    ],
})

# Extract a new column "genre" from "overview" using an LLM.
result = nv.ops.map(
    df,
    user_instruction="From the movie overview, extract the primary genre of the movie as a single word.",
    input_columns=["overview"],
    output_columns=["genre"],
)
df["genre"] = result.outputs["genre"]
print(df)
```

---

## Configuring the LLM Backbone

Before using any semantic operator, configure the default LLM backbone. Nirvana
currently uses OpenAI-compatible APIs.

```python
import nirvana as nv

nv.configure_llm_backbone(
    model_name="gpt-4.1-mini",
    api_key="<your API key>",
    base_url=None,            # optional: point at any OpenAI-compatible endpoint
)
```

`configure_llm_backbone` is a process-wide setting and only needs to be called once.
You may also set the `OPENAI_API_KEY` environment variable instead of passing
`api_key` explicitly.

---

## Two API Surfaces: Eager and Lazy

Nirvana exposes the same set of semantic operators through two complementary
interfaces:

| Surface | Module | When to use |
|---|---|---|
| **Eager** | `nv.ops.*` | Notebook exploration and ad-hoc analysis. Each call executes immediately and returns a result object. |
| **Lazy**  | `nv.DataFrame.semantic_*` | Production pipelines. Calls record operators in a data lineage DAG that the optimizer can rewrite before execution. |

A single pipeline should use **one** surface. The eager and lazy surfaces are not
designed to be mixed within the same query — `nv.ops.*` expects a `pandas.DataFrame`
while `df.semantic_*` operates on a `nv.DataFrame`.

---

## The Eager API

Eager operators take a `pandas.DataFrame` (or two, in the case of `join`) and return
an output dataclass that contains the result and the LLM cost.

> **Note on event loops.** Each `nv.ops.*` call wraps an internal coroutine with
> `asyncio.run`. To call them from within an existing event loop (Jupyter, FastAPI,
> etc.), apply `nest_asyncio.apply()` first.

### `nv.ops.filter`

Evaluate a natural-language predicate on each row and return a boolean mask.

```python
result = nv.ops.filter(
    df,
    user_instruction="The movie was released after 2000.",
    input_columns=["title", "year"],
)
df_filtered = df[result.output].reset_index(drop=True)
```

**Returns** — `FilterOpOutputs(output: list[bool], cost: float)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_data` | `pd.DataFrame` | required | Input table. |
| `user_instruction` | `str` | required | Natural-language predicate. The LLM is prompted to return `true` or `false` for each row. |
| `input_columns` | `list[str]` | required | Columns the LLM should inspect. May contain one or more column names. |
| `func` | `Callable \| None` | `None` | Optional Python UDF tried before the LLM; falls back to the LLM on exception. |
| `context` | `list[dict] \| str \| None` | `None` | Additional context (for example, few-shot demos). |
| `model` | `str \| None` | `None` | Override the default model for this call. |
| `strategy` | `"plain" \| "fewshot" \| "self-refine"` | `"plain"` | Prompting strategy. `"fewshot"` requires `context`. |
| `limit` | `int \| None` | `None` | Stop after this many rows pass the predicate. |
| `rate_limit` | `int` | `16` | Maximum concurrent LLM calls. |

### `nv.ops.map`

Apply an LLM transformation to each row, producing one or more new columns.

```python
result = nv.ops.map(
    df,
    user_instruction="From the movie overview, extract the primary genre.",
    input_columns=["overview"],
    output_columns=["genre"],
)
df["genre"] = result.outputs["genre"]
```

`map` can produce multiple columns in a single call:

```python
result = nv.ops.map(
    df,
    user_instruction="From the passage, extract the entity name and its nationality.",
    input_columns=["text"],
    output_columns=["entity", "nationality"],
)
df["entity"]      = result.outputs["entity"]
df["nationality"] = result.outputs["nationality"]
```

**Returns** — `MapOpOutputs(outputs: dict[str, list], cost: float)`

The `outputs` dict maps each entry of `output_columns` to a list of per-row values.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_data` | `pd.DataFrame` | required | Input table. |
| `user_instruction` | `str` | required | Natural-language transformation. |
| `input_columns` | `list[str]` | required | Source columns the LLM may read. |
| `output_columns` | `list[str]` | required | New column names to produce. |
| `func` | `Callable \| None` | `None` | Optional Python UDF tried before the LLM. |
| `context` | `list[dict] \| str \| None` | `None` | Additional context. |
| `model` | `str \| None` | `None` | Override the default model. |
| `strategy` | `"plain" \| "fewshot" \| "self-refine"` | `"plain"` | Prompting strategy. |
| `limit` | `int \| None` | `None` | Maximum number of rows to process. |
| `rate_limit` | `int` | `16` | Maximum concurrent LLM calls. |

### `nv.ops.reduce`

Aggregate the values of a single column into one result.

```python
result = nv.ops.reduce(
    df,
    user_instruction="Summarize the common themes across these movie overviews.",
    input_column="overview",
)
print(result.output)
```

**Returns** — `ReduceOpOutputs(output: Any, cost: float)`

The `output` is typically a string but may be `None` if the LLM cannot produce a
result. Always guard against `None` in downstream code.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_data` | `pd.DataFrame` | required | Input table. |
| `user_instruction` | `str` | required | Natural-language aggregation instruction. |
| `input_column` | `str` | required | Single column name to aggregate over. |
| `context` | `list[dict] \| str \| None` | `None` | Additional context. |
| `model` | `str \| None` | `None` | Override the default model. |
| `func` | `Callable \| None` | `None` | Optional Python UDF. |
| `strategy` | `"plain"` | `"plain"` | Currently only `"plain"` is supported. |
| `rate_limit` | `int` | `16` | Maximum concurrent LLM calls. |

> The current `reduce` implementation does not chunk inputs that exceed the LLM's
> context window. For long aggregations, pre-filter or pre-summarize with `map`
> first.

### `nv.ops.rank`

Sort the rows of a DataFrame by an LLM-evaluated criterion.

```python
result = nv.ops.rank(
    df,
    user_instruction="Rank these movies by their relevance to crime drama.",
    input_column="title",
    descend=True,
)
df_ranked = df.iloc[result.ranked_indices].reset_index(drop=True)
```

**Returns** — `RankOpOutputs(ranking: list[int], ranked_indices: list[int], cost: float)`

- `ranked_indices` lists the original row indices in ranked order.
- `ranking` gives the rank position (1-indexed) of each original row.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_data` | `pd.DataFrame` | required | Input table. |
| `user_instruction` | `str` | required | Natural-language ranking criterion. |
| `input_column` | `str` | required | Single column name to rank over. |
| `descend` | `bool` | `True` | If `True`, the highest-ranked item appears first. |
| `context` | `list[dict] \| str \| None` | `None` | Additional context. |
| `model` | `str \| None` | `None` | Override the default model. |
| `func` | `Callable \| None` | `None` | Optional Python UDF. |
| `strategy` | `"plain"` | `"plain"` | Currently only `"plain"` is supported. |
| `rate_limit` | `int` | `16` | Maximum concurrent LLM calls. |

### `nv.ops.join`

Semantically join two DataFrames using an LLM-evaluated matching condition.

```python
clinical_notes = pd.DataFrame({
    "patient": ["Alice", "Bob"],
    "symptom": ["headache", "persistent cough"],
})
drugs = pd.DataFrame({
    "name": ["Ibuprofen", "Salbutamol"],
    "use": [
        "treats mild to moderate pain, including headaches",
        "treats bronchospasm and chronic obstructive pulmonary disease",
    ],
})

result = nv.ops.join(
    left_data=clinical_notes,
    right_data=drugs,
    user_instruction="Does the drug treat the symptom?",
    left_on="symptom",
    right_on="use",
    how="inner",
)
print(result.join_pairs)
```

**Returns** — `JoinOpOutputs(join_pairs: list[tuple], left_join_keys: list[int], right_join_keys: list[int], cost: float)`

- `join_pairs` is a list of `(left_idx, right_idx)` tuples for matched rows.
- `left_join_keys` and `right_join_keys` are the matched row indices on each side
  and can be used directly with `iloc` to materialize the joined table.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `left_data` | `pd.DataFrame` | required | Left input table. |
| `right_data` | `pd.DataFrame` | required | Right input table. |
| `user_instruction` | `str` | required | Natural-language match predicate. |
| `left_on` | `str` | required | Single column name on the left side. |
| `right_on` | `str` | required | Single column name on the right side. |
| `how` | `"inner" \| "left" \| "right"` | `"inner"` | Join type. |
| `strategy` | `"nest" \| "block"` | `"nest"` | `"nest"` evaluates every left-right pair (`O(n × m)`). `"block"` batches pairs to reduce LLM calls. |
| `batch_size` | `int` | `5` | Batch size used when `strategy="block"`. |
| `context` | `list[dict] \| str \| None` | `None` | Additional context. |
| `model` | `str \| None` | `None` | Override the default model. |
| `func` | `Callable \| None` | `None` | Optional Python UDF. |
| `limit` | `int \| None` | `None` | Maximum matches to return. |
| `rate_limit` | `int` | `16` | Maximum concurrent LLM calls. |

### Eager API summary

| Operator | Required column argument | Output dataclass | Result field(s) |
|---|---|---|---|
| `filter` | `input_columns: list[str]` | `FilterOpOutputs` | `output: list[bool]` |
| `map`    | `input_columns: list[str]`, `output_columns: list[str]` | `MapOpOutputs` | `outputs: dict[str, list]` |
| `reduce` | `input_column: str` | `ReduceOpOutputs` | `output: Any` |
| `rank`   | `input_column: str` | `RankOpOutputs` | `ranking, ranked_indices: list[int]` |
| `join`   | `left_on: str`, `right_on: str` | `JoinOpOutputs` | `join_pairs, left_join_keys, right_join_keys` |

`filter` and `map` accept a list of input columns because the LLM may inspect
multiple fields per row. `reduce` and `rank` operate on a single column at a time
and take a singular `input_column` string. `join` matches one column from each side.

---

## The Lazy API

The lazy API wraps a pandas DataFrame in an `nv.DataFrame` object that records each
operator as a node in a data lineage DAG. No LLM calls are issued until you invoke
`optimize_and_execute()`, which lets the optimizer rewrite the plan first.

### Creating a `nv.DataFrame`

```python
import pandas as pd
import nirvana as nv

# From a dict
df = nv.DataFrame({"title": ["The Godfather"], "rating": [9.2]})

# From a pandas DataFrame
df = nv.DataFrame(pd.read_csv("/path/to/movies.csv"))

# From an external file
df = nv.DataFrame.from_external_file("/path/to/movies.csv", sep=",")
```

### Building a lazy plan

Each `semantic_*` method appends a node to the plan and returns `self`, so you can
either chain calls or use separate statements — both styles work:

```python
df = nv.DataFrame(pd.read_csv("/path/to/movies.csv"))

df.semantic_map(
    user_instruction="From the movie overview, extract the primary genre.",
    input_columns=["Overview"],
    output_columns=["Genre"],
)
df.semantic_filter(
    user_instruction="The rating is higher than 8.",
    input_columns=["IMDB_Rating"],
)
df.semantic_filter(
    user_instruction="The movie is a crime movie.",
    input_columns=["Genre"],
)
df.semantic_reduce(
    user_instruction="Summarize the common themes across these crime movies.",
    input_column="Overview",
)
```

The same singular-vs-plural convention as the eager API applies: `semantic_map`
and `semantic_filter` take `input_columns: list[str]`, while `semantic_reduce`,
`semantic_rank`, and `semantic_join` keys each take a single column name as a
string.

#### Lazy method signatures

```python
df.semantic_map(
    user_instruction: str,
    input_columns:  list[str],
    output_columns: list[str],
    context:    list[dict] | str | None = None,
    model:      str | None = None,
    func:       Callable | None = None,
    strategy:   "plain" | "fewshot" | "self-refine" = "plain",
    limit:      int | None = None,
    rate_limit: int = 16,
) -> nv.DataFrame
```

```python
df.semantic_filter(
    user_instruction: str,
    input_columns:  list[str],
    func:       Callable | None = None,
    context:    list[dict] | str | None = None,
    model:      str | None = None,
    strategy:   "plain" | "fewshot" | "self-refine" = "plain",
    limit:      int | None = None,
    rate_limit: int = 16,
) -> nv.DataFrame
```

```python
df.semantic_reduce(
    user_instruction: str,
    input_column: str,
    context:    list[dict] | str | None = None,
    model:      str | None = None,
    func:       Callable | None = None,
    strategy:   "plain" = "plain",
    rate_limit: int = 16,
) -> nv.DataFrame
```

```python
df.semantic_rank(
    user_instruction: str,
    input_column: str,
    descend:    bool = True,
    context:    list[dict] | str | None = None,
    model:      str | None = None,
    func:       Callable | None = None,
    strategy:   "plain" = "plain",
    limit:      int | None = None,
    rate_limit: int = 16,
) -> nv.DataFrame
```

```python
df.semantic_join(
    other:      nv.DataFrame,
    user_instruction: str,
    left_on:    str,
    right_on:   str,
    how:        "inner" | "left" | "right" = "inner",
    context:    list[dict] | str | None = None,
    model:      str | None = None,
    func:       Callable | None = None,
    strategy:   "nest" | "block" = "nest",
    limit:      int | None = None,
    rate_limit: int = 16,
    batch_size: int = 5,
) -> nv.DataFrame
```

### Triggering execution

`optimize_and_execute` runs the configured optimizations on the lineage and then
executes the plan:

```python
config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    do_physical_optimization=False,
)
output, cost, runtime = df.optimize_and_execute(optim_config=config)
```

It returns a 3-tuple:

| Element | Type | Description |
|---|---|---|
| `output` | `nv.DataFrame` | The optimized DataFrame containing the result of the plan. |
| `cost` | `float` | Total LLM cost (USD) across all executed operators. |
| `runtime` | `float` | Wall-clock execution time in seconds, including any optimization overhead. |

If `optim_config` is omitted, the default `OptimizeConfig()` is used (both logical
and physical optimization enabled).

---

## Query Optimization

Nirvana separates query optimization into two phases, inspired by the architecture
of relational systems such as Spark SQL:

1. **Logical plan optimization** rewrites the operator graph using transformation
   rules to reduce the number of LLM invocations while preserving result quality.
2. **Physical plan optimization** assigns the most cost-effective LLM backend to
   each operator using a sample-based improvement-score metric.

Both phases are controlled through `nv.optim.OptimizeConfig`.

### `OptimizeConfig`

```python
nv.optim.OptimizeConfig(
    # master switches
    do_logical_optimization:  bool = True,
    do_physical_optimization: bool = True,

    # sampling and iteration budgets
    sample_size:    int   = 5,
    max_rounds:     int   = 5,
    improve_margin: float = 0.2,

    # logical transformation rules (independent toggles)
    filter_pullup:    bool = True,
    filter_pushdown:  bool = True,
    map_pullup:       bool = True,
    non_llm_pushdown: bool = True,
    non_llm_replace:  bool = True,
    operator_fusion:  bool = True,

    # candidate models for physical optimization
    available_models: list[str] = [],
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `do_logical_optimization` | `bool` | `True` | Enable the logical plan optimizer. |
| `do_physical_optimization` | `bool` | `True` | Enable the physical plan optimizer. |
| `sample_size` | `int` | `5` | Number of records sampled for plan evaluation. |
| `max_rounds` | `int` | `5` | Maximum rewrite iterations during logical optimization. |
| `improve_margin` | `float` | `0.2` | Minimum relative improvement required to swap a model in physical optimization. |
| `filter_pullup` | `bool` | `True` | Enable the filter pull-up rule. |
| `filter_pushdown` | `bool` | `True` | Enable the filter push-down rule. |
| `map_pullup` | `bool` | `True` | Enable the map pull-up rule. |
| `non_llm_pushdown` | `bool` | `True` | Enable pushing deterministic predicates ahead of LLM operators. |
| `non_llm_replace` | `bool` | `True` | Enable replacing LLM operators with equivalent Python UDFs when possible. |
| `operator_fusion` | `bool` | `True` | Enable merging adjacent operators on the same column into a single LLM call. |
| `available_models` | `list[str]` | `[]` | Candidate model names available for physical optimization. |

### Logical plan optimization

Disable the physical optimizer to apply only logical rewrites. This is the cheapest
useful configuration: the logical optimizer reduces LLM calls without per-operator
sampling overhead.

```python
config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    do_physical_optimization=False,
)
output, cost, runtime = df.optimize_and_execute(optim_config=config)
```

Individual transformation rules can be turned off for ablation studies:

```python
config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    non_llm_replace=False,         # disable non-LLM replacement
)
```

#### Transformation rules

| Rule | Description |
|---|---|
| **Filter pushdown** | Move filters earlier in the plan so that downstream operators see fewer rows. |
| **Filter pullup** | Move filters later when an upstream operator already produces the necessary input — useful when reordering reduces total LLM calls. |
| **Map pullup** | Defer expensive `map` operators until after filtering. |
| **Non-LLM pushdown** | Push purely deterministic predicates (for example, numeric comparisons) ahead of LLM operators using compiled UDFs. |
| **Non-LLM replacement** | Replace an entire LLM-backed operator with an LLM-generated Python function when the instruction can be expressed as deterministic code. |
| **Operator fusion** | Merge adjacent operators on the same column into a single LLM call (for example, `rating > 8.5` AND `rating < 9` become one filter). |

> **Authoring tip**: when writing lazy plans you do not need to manually combine
> related conditions. Express each condition as its own `semantic_filter` call —
> the operator-fusion rule will merge them. Verbose plans give the optimizer more
> degrees of freedom.

### Physical plan optimization

Enable physical optimization to let Nirvana select the most cost-effective LLM for
each operator. Populate `available_models` with the candidate model names you want
considered.

```python
config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    do_physical_optimization=True,
    sample_size=5,
    improve_margin=0.2,
    available_models=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"],
)
output, cost, runtime = df.optimize_and_execute(optim_config=config)
```

Physical optimization runs each candidate model on a sample of `sample_size` rows
per operator and computes an improvement score. A model is selected when its score
exceeds the current best by at least `improve_margin`. See Section 4 of the Nirvana
paper for the algorithm and its theoretical guarantees.

Physical optimization adds significant overhead (typically several times the LLM
calls of plain execution) because it must evaluate multiple candidate models on
the sample. For benchmarking, start with `do_physical_optimization=False` to measure
the contribution of logical optimization in isolation, and then enable it to measure
the additional gains.

---

## Multi-Modal Data

Nirvana ships pandas extension dtypes for image, audio, and file columns:

| Type | Class | Module |
|---|---|---|
| Image | `nv.ImageDtype`, `nv.ImageArray` | `nirvana.dataframe.arrays.image` |
| Audio | `nv.AudioDtype`, `nv.AudioArray` | `nirvana.dataframe.arrays.audio` |
| File  | `nv.FileDtype`,  `nv.FileArray`  | `nirvana.dataframe.arrays.file`  |

Once a column is cast to one of these dtypes, all semantic operators automatically
detect it and route the value into the LLM as the appropriate multi-modal payload
(for example, an `input_image` content block when calling OpenAI's Responses API).

### Image columns

`ImageArray` accepts the following input forms and normalizes them to base64
data URIs:

| Input | Treatment |
|---|---|
| `PIL.Image.Image` | Encoded as PNG and prefixed with `data:image/png;base64,` |
| `bytes` | Wrapped with `data:image/png;base64,` |
| `str` starting with `data:image` | Returned as-is |
| `str` starting with `https://` | Returned as-is (the URL is passed to the LLM directly) |
| `str` starting with `s3://` | Fetched via `boto3` and re-encoded as a data URI |
| Any other `str` | Treated as a local file path, opened, and base64-encoded |

#### Eager example

```python
import pandas as pd
import nirvana as nv

logo_imgs = nv.ImageArray([
    "https://spark.apache.org/images/spark-logo.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/488px-PyTorch_logo_black.svg.png",
])
df = pd.DataFrame({
    "name":  ["Spark", "PyTorch"],
    "logos": logo_imgs,
})

result = nv.ops.filter(
    df,
    user_instruction="Is this image a software logo?",
    input_columns=["logos"],
)
df_filtered = df[result.output].reset_index(drop=True)
```

#### Lazy example

```python
import pandas as pd
import nirvana as nv

logo_imgs = nv.ImageArray([
    "https://spark.apache.org/images/spark-logo.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/488px-PyTorch_logo_black.svg.png",
])
df = nv.DataFrame(pd.DataFrame({
    "name":  ["Spark", "PyTorch"],
    "logos": logo_imgs,
}))

df.semantic_filter(
    user_instruction="Is this image a software logo?",
    input_columns=["logos"],
)

config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    do_physical_optimization=True,
)
output, cost, runtime = df.optimize_and_execute(optim_config=config)
```

`AudioDtype` and `FileDtype` follow the same pattern. See `nirvana/dataframe/arrays/`
for the supported input forms of each.

---

## End-to-End Example

The following example combines several operators on a movie dataset and demonstrates
both the lazy API and logical optimization. It is also available as Listing 1 in the
Nirvana paper.

```python
import pandas as pd
import nirvana as nv

nv.configure_llm_backbone(model_name="gpt-4.1-mini", api_key="<your API key>")

df = nv.DataFrame(pd.read_csv("/path/to/movies.csv"))

# 1. Extract a derived column from a free-text field.
df.semantic_map(
    user_instruction="From the movie overview, extract the primary genre.",
    input_columns=["Overview"],
    output_columns=["Genre"],
)

# 2. Two filter conditions on the rating column. The optimizer will fuse them
#    into a single LLM call via the operator-fusion rule, or replace them
#    entirely with a Python UDF via the non-LLM-replacement rule.
df.semantic_filter(
    user_instruction="The rating is higher than 8.5.",
    input_columns=["IMDB_Rating"],
)
df.semantic_filter(
    user_instruction="The rating is lower than 9.",
    input_columns=["IMDB_Rating"],
)

# 3. Filter on the derived genre column.
df.semantic_filter(
    user_instruction="The movie belongs to the crime genre.",
    input_columns=["Genre"],
)

# 4. Aggregate the surviving rows.
df.semantic_reduce(
    user_instruction="Summarize the common plot structure of these high-rated crime movies.",
    input_column="Overview",
)

config = nv.optim.OptimizeConfig(
    do_logical_optimization=True,
    do_physical_optimization=True,
    sample_size=5,
    improve_margin=0.2,
    available_models=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"],
)
output, cost, runtime = df.optimize_and_execute(optim_config=config)

print(f"cost = ${cost:.4f}, runtime = {runtime:.2f}s")
```

---

## Citation

If you use Nirvana in your research, please cite:

```bibtex
@article{zhu2025nirvana,
  title   = {Beyond Relational: Semantic-Aware Multi-Modal Analytics with LLM-Native Query Optimization},
  author  = {Zhu, Junhao and Chen, Lu and Ke, Xiangyu and Fang, Ziquan and Li, Tianyi and Gao, Yunjun and Jensen, Christian S.},
  journal = {arXiv preprint arXiv:2511.19830},
  year    = {2025},
}
```

---

## License

MIT. See [LICENSE](LICENSE) for details.
