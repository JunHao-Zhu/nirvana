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
>>> df = mjg.DataFrame(pd.read_csv("/testdata/imdb_top_1000.csv").sample(n=200).drop("Genre", axis=1))
>>> df.semantic_map(user_instruction="According to the movie overview, extract the genre of each movie.", input_column="Overview", output_column="Genre")
>>> df.semantic_filter(user_instruction="The rating is higher than 8.5.", input_column="IMDB_Rating")
>>> df.semantic_filter(user_instruction="The rating is lower than 9.", input_column="IMDB_Rating")
>>> df.semantic_filter(user_instruction="The movie belongs to crime movies", input_column="Genre")
>>> df.semantic_reduce(user_instruction="Find the maximum rating in the rest movies.", input_column="Genre")
```
The initial logical plan and its cost are like, 
```python
>>> df.print_logical_plan()
map: [Overview]->[Genre] (According to the movie overview, extract the genre of each movie.) =>
filter: [IMDB_Rating]->[Bool] (The rating is higher than 8.5.) =>
filter: [IMDB_Rating]->[Bool] (The rating is lower than 9.) =>
filter: [Genre]->[Bool] (The movie belongs to crime movies.) =>
reduce: [Genre] (Find the maximum rating in the rest movies.)
```
After logical plan optimization, the new logical plan and its cost are like,
```python
>>> res_after_optim, token_cost_after_optim, time_after_optim = df.optimize_and_execute()
Plan optimization is finished, taking 10.13 seconds. Here are some statistics:
initial plan cost: 160 -> optimized plan cost: 24.75
initial plan accuracy: 1.0 -> optimized plan accuracy: 1.0
>>> df.print_logical_plan()
filter: [IMDB_Rating]->[Bool] (The rating is higher than 7 and lower than 8.) =>
map: [Overview]->[Genre] (According to the movie overview, extract the genre of each movie.) =>
filter: [Genre]->[Bool] (The movie belongs to crime movies.) =>
reduce: [Genre] (Find the maximum rating in the rest movies.)
>>> token_cost_after_optim
8248
>>> time_after_optim
88.65716409683228
```
Now support three logical plan optimization rules: `Operator fusion`, `Filter pushdown`, and `Non-LLM replacement`.

As a comparison, Palimpzest takes 422 seconds finishing the same query and generates low-quality results.

### Physical plan optimization
The physical plan optimizer allocates the most cost-effective model to each operator in the logical plan.

Here is a usage example. First define the optimization config, then optimize and execute the query.
```python
from mahjong.optim import OptimizeConfig
>>> config = OptimizeConfig(do_logical_optimization=False, do_physical_optimization=True, sample_size=10, improve_margin=0.2, approx_mode=True)
>>> df.optimize_and_execute(config=config)
```
