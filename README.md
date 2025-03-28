# Mahjong: LLM-powered Multi-modal Native Data Lake

from zero to hero

## TODO
1. build dataframe, Pandas is a good start like [LOTUS](https://github.com/lotus-data/lotus).
2. build semantic operators by referencing [LOTUS](https://github.com/lotus-data/lotus) and [DocETL](https://github.com/ucbepic/docetl).


## Operators
The coding example of operators is as follow:

`Map`: perform a projecton on target data based on a predicate (the code refers to ops/map.py (execution) and prompt_templates/map_prompter.py (prompts))
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
... ]
... })
>>> map.execute(df["overview"], "According to the movie overview, extract the genre of each movie", target_schema = "genre", strategy="plain_llm")

MapOpOutput(
    field_name = "genre",
    output = ["crime, drama", "action, thriller, superhero"]
)
```