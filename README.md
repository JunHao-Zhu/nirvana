# Mahjong: Extend Data Analytics to Multi-modal Data with Agentic Orchestration

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
>>> mjg.ops.map(df, "According to the movie overview, extract the genre of each movie.", input_column="overview", output_column = "genre", strategy="plain_llm")
MapOpOutputs(
    field_name = "genre",
    output = ["crime, drama", "action, thriller, superhero"]
)
```

Operator `filter`: Evaluate a condition on the target data (returning either True or False) (the code refers to ops/filter.py (execution) and prompt_templates/filter_prompter.py (prompts))
```python
>>> mjg.ops.filter(df, "Whether the movie is released after 2000?", input_column="title", strategy="plain_llm")
FilterOpOutputs(
    output = [False, True]
)
```

Operator `reduce`: Aggregate a set of data based on the user instruction (the code refers to ops/reduce.py (execution) and prompt_templates/reduce_prompter.py (prompts))
```python
>>> mjg.ops.reduce(df, "Based on the overviews of the given movies, provide several common points of these movies. The common points should be concise.", input_column="overview")
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
>>> df = mjg.DataFrame(df)
>>> df.semantic_filter("Whether the image is a software logo?", input_schema="logos", strategy="plain_llm")
>>> df.optimize_and_execute()
```
For example, `semantic_filter` creates an operator node in the data lineage of dataframe `df`, and `optimize_and_execute()` optimizes the logical plan and physical plan and executes the query processing.

Data lineage includes two core abstrations: `LineageOpNode` and `LineageDataNode`. `LineageOpNode` serves the operator optimization and execution. `LineageDataNode` serves the optimizations from perspective of data, like materializing the intermediate result and reusing it. By default, no `LineageDataNode` is included in data lineage, and it is created by setting `materialize` parameter `True` when necessary.
```python
>>> df.semantic_filter("Whether the image is a software logo?", input_column="logos", strategy="plain_llm")
```

### Logical plan optimization
An agentic optimization workflow starts after the initial logical plan given and an optimize task
invoked. A usage example of logical plan optimization is shown as follow.
```python
>>> df = mjh.DataFrame(pd.read_csv("/testdata/imdb_top_1000.csv").sample(n=200).drop("Genre", axis=1))
>>> df.semantic_map(user_instruction="According to the movie overview, extract the genre of each movie.", input_column="Overview", output_column="Genre")
>>> df.semantic_filter(user_instruction="The rating is higher than 7.", input_column="IMDB_Rating")
>>> df.semantic_filter(user_instruction="The rating is lower than 8.", input_column="IMDB_Rating")
>>> df.semantic_reduce(user_instruction="Count the number of crime movies.", input_column="Genre")
```
The initial logical plan and its cost are like, 
```python
>>> df.print_logical_plan()
map: [Overview]->[Genre] (According to the movie overview, extract the genre of each movie.) =>
filter: [IMDB_Rating]->[Bool] (The rating is higher than 7.) =>
filter: [IMDB_Rating]->[Bool] (The rating is lower than 8.) =>
reduce: [Genre] (Count the number of crime movies.)
>>> res_before_optim, token_cost_before_optim = df.tile.execute(df)
>>> print(token_cost_before_optim)
84542
```
After logical plan optimization, the new logical plan and its cost are like,
```python
>>> res_after_optim, token_cost_after_optim = df.tile.optimize_and_optimize()
Plan optimization is finished, here are some statistics:
initial plan cost: 4289 -> optimized plan cost: 2572
initial plan accuracy: 1.0 -> optimized plan accuracy: 1.0
>>> df.print_logical_plan()
filter: [IMDB_Rating]->[Bool] (The rating is higher than 7 and lower than 8.) =>
map: [Overview]->[Genre] (According to the movie overview, extract the genre of each movie.) =>
reduce: [Genre] (Count the number of crime movies.)
>>> print(token_cost_after_optim)
46074
```
Now support two logical plan optimization rules: `Operator fusion` and `Filter pushdown`.
