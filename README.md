# Nirvana
<!--- BADGES: START --->
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2511.19830)
[![PyPI](https://img.shields.io/pypi/v/nirvana-ai)](https://pypi.org/project/nirvana-ai/)
[![Documentation](https://img.shields.io/badge/Documentation-docs-green)](https://JunHao-Zhu.github.io/nirvana)
<!--- BADGES: END --->

An LLM-powered semantic data analytics programming framework.

> 📖 **Documentation**: Full documentation is available at [docs/](docs/) or build locally with `mkdocs serve`.

## Tutorial

### Installation
```bash
pip install nirvana-ai
```
To install the latest version from `main`:
```bash
pip install git+https://github.com/JunHao-Zhu/nirvana.git
```

### LLM backbone configuration
Before using semantic operators, first configure the llm settings used in the system. Taking OpenAI as an example,
```python
>>> import nirvana as nv
>>> nv.configure_llm_backbone(
...     model_name = "gpt-4.1", 
...     api_key = "<Your API Key>",
...     base_url = None
... )
```

### Operators
Operator `map`: Perform a projecton on the target data based on a predicate (the code refers to ops/map.py (execution) and prompt_templates/map_prompter.py (prompts))
```python
>>> import pandas as pd
>>> import nirvana as nv

>>> df = pd.DataFrame(
... {
...     "title": ["The Godfather", "The Dark Knight"], 
...     "overview": [
...         "An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.", 
...         "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
...     ]
... })
>>> nv.ops.map(df, "According to the movie overview, extract the genre of each movie.", input_column="overview", output_columns = ["genre"], strategy="plain")
MapOpOutputs(
    field_name = ["genre"],
    output = {"genre": ["crime, drama", "action, thriller, superhero"]}
)
```

Operator `filter`: Evaluate a condition on the target data (returning either True or False) (the code refers to ops/filter.py (execution) and prompt_templates/filter_prompter.py (prompts))
```python
>>> nv.ops.filter(df, "Whether the movie is released after 2000?", input_column="title", strategy="plain")
FilterOpOutputs(
    output = [False, True]
)
```

Operator `reduce`: Aggregate a set of data based on the user instruction (the code refers to ops/reduce.py (execution) and prompt_templates/reduce_prompter.py (prompts))
```python
>>> nv.ops.reduce(df, "Based on the overviews of the given movies, provide several common points of these movies. The common points should be concise.", input_column="overview")
ReduceOpOutputs(
    output = "1. Both movies involve a transfer or test of leadership and capability. 2. The protagonists face significant psychological challenges. 3. The stories revolve around crime and justice. 4. The main characters are reluctant or tested in their roles. 5. Both narratives feature a clash between order and chaos."
)
```
> The current version of the reduce operator is simple. In the next step, we will implement it using several optimizations, like `summarize and aggregate` and `incremental aggregation`.

Operator `rank`: Rank a set of data based on the user instruction by quicksort (under intensive testing)
```python
>>> nv.ops.rank(df, "rank the movies by their relevance to DC Comics.", input_column="title")
RankOpOutputs(
    output = [2, 1]
)
```

Operator `join`: Join values of two columns against a specific user's instruction (under intensive testing). The current implementation has inherent quadratic complexity which we aim to avoid.
```python
# left_data: clinical_note
# | name | age | gender | symptom |
# | Alice | 12 | F | headache |
# | Bob | 20 | M | have a cough |

# right_data: drug
# | name | medical_use |
# | Salbutamol | treat bronchospasm, as well as chronic obstructive pulmonary disease |
# | ibuprofen | treat mild to moderate pain, painful menstruation, osteoarthritis, dental pain, headaches, and pain from kidney stones|
>>> nv.ops.join(left_data=clinical_note, right_data=drug, user_instruction="Does the drug cure the possible disease according to the symptoms?", left_on="symptom", right_on="medical_use", how="inner")
RankOpOutputs(
    joined_pairs = [("headache", "ibuprofen"), ("have a cough", "Salbutamol")],
    left_join_keys = [1, 2],
    right_join_keys = [2, 1]
)
```
For now, it supports inner join, left join, and right join by setting parameter `how` to `inner`, `left`, and `right`.

Operator `discover`: Discover relevant data from a data lake based on a query. This operation is applied to a data lake with an interface of DataLake class above it (see datalake/datalake.py)

### Data lineage
Data lineage (a directed acyclic graph) enables lazy exeuction and logical and physical plan optimizations. Data lineage is created along with dataframe.
```python
>>> logo_imgs = nv.ImageArray([  # Image data
...     "https://spark.apache.org/images/spark-logo.png",
...     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/488px-PyTorch_logo_black.svg.png",
...     "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Databricks_Logo.png/960px-Databricks_Logo.png"
... ])
>>> df = pd.DataFrame({
...     "names": ["Spark", "Pytorch", "Databricks"],
...     "logos": logo_imgs
... })
>>> df = nv.DataFrame(df)
>>> df.semantic_filter("Whether the image is a software logo?", input_schema="logos", strategy="plain_llm")
>>> config = nv.optim.OptimizeConfig(do_logical_optimization=True, do_physical_optimization=True)
>>> df.optimize_and_execute(optim_config=config)
```
For example, with `semantic_filter`, it creates an op node in the data lineage of dataframe `df`, and `optimize_and_execute()` optimizes the logical plan and physical plan, then executes the query processing.

The core abstractions of data lineage is `LineageNode` (found in `lineage/abstractions.py`), serving the operator optimization and execution. The usages of `LineageNode` are found in `lineage/mixin.py`.

### Plan optimization
Considering the following semantic data analytics query,
```python
>>> df = nv.DataFrame(pd.read_csv("/testdata/movie_data.csv"))
>>> df.semantic_map(user_instruction="According to the movie overview, extract the genre of each movie.", input_column="Overview", output_column="Genre")
>>> df.semantic_filter(user_instruction="The rating is higher than 7.", input_column="IMDB_Rating")
>>> df.semantic_filter(user_instruction="The rating is lower than 8.", input_column="IMDB_Rating")
>>> df.semantic_reduce(user_instruction="Count the number of crime movies.", input_column="Genre")
```
Nirvana does logical optimization and physical optimization separately. You can turn on/off each optimization in `OptimizeConfig` and execute the query processing with the configure.
```python
>>> config = nv.optim.OptimizeConfig(do_logical_optimization=True, do_physical_optimization=False)
>>> output, cost, runtime = df.optimize_and_execute(optim_config=config)
```

#### Logical plan optimization
Rule-based logical plan optimization is adopted in Nirvana. Now we support 5 transformation rules (found in `optim/rules.py`):
- `Non-llm replacement`: Replaces NL instructions over non-image/video/audio data with an equivalent compute function (powered by LLMs).
- `Map pullup`: Pulls up maps to the top of the query plan.
- `Filter pullup`: Identifies cases where a filter can be applied on columns in other tables.
- `Filter pushdown`: Pushes filters down into the query plan and duplicates filters over equivalency sets.
- `Non-llm pushdown`: Pushes operators using non-llm functions/tools down into the query plan.

The rules are applied in a sequential order as they are listed here. The knob to turn on/off each rule is defined in `OptimizeConfig`, and by default all the rules will be used. If you want to turn off the LLM-powered semantic transformation rule `Non-llm replacement`, for example, you can do this:
```python
>>> config = nv.optim.OptimizeConfig(do_logical_optimization=True, non_llm_replace=False)
```

#### Physical plan optimization
In this version, physical plan optimization assigns the most cost-effective model to each operator in a query plan (found in `optim/physical.py`). To use it, you need to set `do_physical_optimization` to `True` and set hyperparameters like `sample_size` and `improve_margin` in `OptimizeConfig`.
```python
>>> config = nv.optim.OptimizeConfig(do_logical_optimization=True, do_physical_optimization=True, sample_size=5, improve_margin=0.2)
```
