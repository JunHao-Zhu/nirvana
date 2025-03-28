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
Operator `map`: Perform a projecton on target data based on a predicate (the code refers to ops/map.py (execution) and prompt_templates/map_prompter.py (prompts))
```python
>>> import pandas as pd
>>> import mahjong as mjg
>>> from mahjong.ops.map import MapOperation as map

>>> df = pd.DataFrame(
... {
...     "title": ["The Godfather", "The Dark Knight"], 
...     "overview": [
...         "The story follows the Corleone crime family, ...", 
...         "The story follows Batman, Police Lieutenant Jim Gordon (Gary Oldman), ..."
...     ]
... })
>>> map.execute(df["overview"], "According to the movie overview, extract the genre of each movie", target_schema = "genre", strategy="plain_llm")
MapOpOutput(
    field_name = "genre",
    output = ["crime, drama", "action, thriller, superhero"]
)
```

Operator `filter`: