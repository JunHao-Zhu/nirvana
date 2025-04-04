# Mahjong: Revolutionize Multi-modal Data Analytics with LLMs

From zero to hero

## Tutorial

### LLM backbone configuration
Before using semantic operators, first configure the llm settings used in the system. Taking DeepSeek as an example,
```python
>>> import mahjong as mjg
>>> mjg.configure_llm_backbone(
...     model_name = "deepseek-chat", 
...     api_key = "<Your API Key>",
...     base_url = "https://api.deepseek.com"
... )
```

### Operators
Operator `map`: Perform a projecton on the target data based on a predicate (the code refers to ops/map.py (execution) and prompt_templates/map_prompter.py (prompts))
```python
>>> import pandas as pd
>>> import mahjong as mjg

>>> df = pd.DataFrame(
... {
...     "title": ["The Godfather", "The Dark Knight"], 
...     "overview": [
...         "An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.", 
...         "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
...     ]
... })
>>> mjg.ops.map(df, "According to the movie overview, extract the genre of each movie.", input_schema="overview", output_schema = "genre", strategy="plain_llm")
MapOpOutputs(
    field_name = "genre",
    output = ["crime, drama", "action, thriller, superhero"]
)
```

Operator `filter`: Evaluate a condition on the target data (returning either True or False) (the code refers to ops/filter.py (execution) and prompt_templates/filter_prompter.py (prompts))
```python
>>> mjg.ops.filter(df, "Whether the movie is released after 2000?", input_schema="title", strategy="plain_llm")
FilterOpOutputs(
    output = [False, True]
)
```

Operator `reduce`: Aggregate a set of data based on the user instruction (the code refers to ops/reduce.py (execution) and prompt_templates/reduce_prompter.py (prompts))
```python
>>> mjg.ops.reduce(df, "Based on the overviews of the given movies, provide several common points of these movies. The common points should be concise.", input_schema="overview")
ReduceOpOutputs(
    output = "1. Both movies involve a transfer or test of leadership and capability. 2. The protagonists face significant psychological challenges. 3. The stories revolve around crime and justice. 4. The main characters are reluctant or tested in their roles. 5. Both narratives feature a clash between order and chaos."
)
```
> The current version of the reduce operator is simple. In the next step, we will implement it using several optimizations, like `summarize and aggregate` and `incremental aggregation`.

### Data lineage
Data lineage (a directed acyclic graph) enables lazy exeuction and logical and physical plan optimizations. Data lineage is created along with dataframe.
```python
>>> logo_imgs = mjg.ImageArray([  # Image data
...     "https://spark.apache.org/images/spark-logo.png",
...     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/488px-PyTorch_logo_black.svg.png",
...     "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Databricks_Logo.png/960px-Databricks_Logo.png"
... ])
>>> df = pd.DataFrame({
...     "names": ["Spark", "Pytorch", "Databricks"],
...     "logos": logo_imgs
... })
>>> df.tile.semantic_filter("Whether the image is a software logo?", input_schema="logos", strategy="plain_llm")
>>> df.tile.execute()
```
For example, `semantic_filter` creates an operator node in the data lineage of dataframe `df`, and `execute()` optimizes the logical plan and physical plan and executes the query processing.

Data lineage includes two core abstrations: `LineageOpNode` and `LineageDataNode`. `LineageOpNode` serves the operator optimization and execution. `LineageDataNode` serves the optimizations from perspective of data, like materializing the intermediate result and reusing it. By default, no `LineageDataNode` is included in data lineage, and it is created by setting `materialize` parameter `True` when necessary.
```python
>>> df.tile.semantic_filter("Whether the image is a software logo?", input_schema="logos", strategy="plain_llm", materialize=True)
```

## TODO
- [x] Build dataframe that supports image data. Pandas is a good start like [LOTUS](https://github.com/lotus-data/lotus).
- [ ] Build semantic operators by referencing [LOTUS](https://github.com/lotus-data/lotus) and [DocETL](https://github.com/ucbepic/docetl).
- [ ] Build our dataframe independent of Pandas, like [Meerkat](https://github.com/HazyResearch/meerkat).
- [ ] Optimizations for operators and operator orchastration (OptimizerMixin, LineageMixin for DataFrame):
    <ol start="0">
    <li>[ ] Determine the optimization targets (trade off between latency, cost, and accuracy).</li>
    <li>[ ] Add data lineage recording the operators (and NL instructions) for operator fusion.</li>
    <li>[ ] Add optimizers choosing an appropriate physical execution for each op.</li>
    </ol>
- [ ] Optimizations for connector to external databases and data lakes.
