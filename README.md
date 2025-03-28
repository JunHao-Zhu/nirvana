# Mahjong: LLM-powered Multi-modal Native Data Lake

from zero to hero

## TODO
1. build dataframe, Pandas is a good start like [LOTUS](https://github.com/lotus-data/lotus).
2. build semantic operators by referencing [LOTUS](https://github.com/lotus-data/lotus) and [DocETL](https://github.com/ucbepic/docetl).


## Tutorial

### LLM Backbone configuration
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
>>> mjg.ops.map(df["overview"], "According to the movie overview, extract the genre of each movie.", target_schema = "genre", strategy="plain_llm")
MapOpOutput(
    field_name = "genre",
    output = ["crime, drama", "action, thriller, superhero"]
)
```

Operator `filter`: Evaluate a condition on the target data (returning either True or False) (the code refers to ops/filter.py (execution) and prompt_templates/filter_prompter.py (prompts))
```python
>>> mjg.ops.filter(df["title"], "Whether the movie is released after 2000?", strategy="plain_llm")
FilterOpOutput(
    output = [False, True]
)
```