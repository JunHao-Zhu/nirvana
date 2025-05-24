"""
DataFrame class for handling data in a tabular format.
To begin, DataFrame is a combination of regular pandas DataFrame and additional unstructured data types (e.g., text, images, and audio).
Example:
| location | house price | comment | house picture |
|----------|-------------|---------|---------------|
|161 Auburn St. Unit 161, Cambridge, MA 02139| $1,000,000 | "Great location!" | ![house](house.jpg) (a link to the house image) |
|...|...|...|...|

A simple version that directly uses pandas DataFrame and supports image data type through pandas api extension, like lotus.
Later, we will implement an independent DataFrame of pandas DataFrame. The usage will be like:
```python
import mahjong as mjg

df = mjg.DataFrame({
    'location': ['161 Auburn St. Unit 161, Cambridge, MA 02139'],
    'house price': ['$1,000,000'],
    'comment': ['Great location!'],
    'house picture': [mjg.Image('house.jpg')]
})
```
Moreover, in the future, we consider reading data from data lake storages (e.g, S3, Delta Lake, etc.)
"""
from typing import Union
import pandas as pd

from mahjong.lineage.mixin import LineageMixin


class DataFrame(LineageMixin):
    def __init__(
            self,
            data: pd.DataFrame = None,
            *args,
            **kwargs
    ):
        self._data = data
        self.columns = list(data.columns)
        self.last_node = None

    def __len__(self):
        _len = self.nrows
        return _len
    
    def __contains__(self, item):
        return self.columns.__contains__(item)
    
    @property
    def columns(self):
        return list(self._data.columns)
    
    @columns.setter
    def columns(self, columns):
        self._data.columns = columns
    
    @property
    def primary_key(self):
        return self._primary_key
    
    @property
    def nrows(self):
        return self._data.shape[0]
    
    @classmethod
    def from_external_file(cls, path: str, sep=',', **kwargs):
        df = pd.read_table(path, sep=sep, **kwargs)
        return cls(df)

    def semantic_map(self, user_instruction, input_column, output_column):
        self.add_operator(op_name="map",
                          user_instruction=user_instruction,
                          input_column=input_column,
                          output_column=output_column,
                          fields=self.columns)
        
    def semantic_filter(self, user_instruction, input_column):
        self.add_operator(op_name="filter",
                          user_instruction=user_instruction,
                          input_column=input_column,
                          fields=self.columns)
        
    def semantic_reduce(self, user_instruction, input_column):
        self.add_operator(op_name="reduce",
                          user_instruction=user_instruction,
                          input_column=input_column,
                          fields=self.columns)
        
    def optimize_and_execute(self, optim_config = None):
        self.create_plan_optimizer(optim_config)
        if self.optimizer.config.do_logical_optimization:
            self.last_node = self.optimizer.optimize_logical_plan(self.last_node, "df", self.columns)
        if self.optimizer.config.do_physical_optimization:
            output, cost, runtime = self.optimizer.optimize_physical_plan(
                self.last_node,
                self._data,
            )
        else:
            output, cost, runtime = self.execute(self._data)
        return output, cost, runtime
