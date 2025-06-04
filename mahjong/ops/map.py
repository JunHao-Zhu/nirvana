"""
Map: Perform a projecton on target data based on NL predicates
"""
import functools
from typing import Any, Iterable, Callable
from dataclasses import dataclass
import pandas as pd

from mahjong.dataframe.arrays.image import ImageDtype
from mahjong.ops.base import BaseOpOutputs, BaseOperation
from mahjong.ops.prompt_templates.map_prompter import MapPrompter


def map_wrapper(
    input_data: pd.DataFrame, 
    user_instruction: str = None,
    func: Callable = None, 
    input_column: str = None,
    output_column: str = None, 
    strategy: str = None, **kwargs
):
    map_op = MapOperation()
    outputs = map_op.execute(
        input_data=input_data,
        user_instruction=user_instruction,
        input_column=input_column,
        output_column=output_column,
        strategy=strategy,
        **kwargs
    )
    return outputs


@dataclass
class MapOpOutputs(BaseOpOutputs):
    field_name: str = None
    output: Iterable[Any] = None

    def __add__(self, other: "MapOpOutputs"):
        return MapOpOutputs(
            field_name=self.field_name,
            output=self.output + other.output,
            cost=self.cost + other.cost
        )


class MapOperation(BaseOperation):
    """ TODO: Implement MapOperation class """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("map", *args, **kwargs)
        self.prompter = MapPrompter()
    
    def _plain_llm_execute(self, data: Any, user_instruction: str, dtype: str, **kwargs):
        if dtype == "str":
            data = f"{kwargs['field_name']}: {str(data)}"
        full_prompt = self.prompter.generate_prompt(data, user_instruction, dtype)
        output = self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
        return output["output"], output["cost"]

    def _llm_cot_execute(self, data: Any, user_instruction: str, dtype: str, demos, **kwargs):
        if dtype == "str":
            data = f"{kwargs['field_name']}: {str(data)}"
        full_prompt = self.prompter.generate_cot_prompt(data, user_instruction, dtype, demos)
        output = self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
        return output["output"], output["cost"]

    def execute(
            self, 
            input_data: pd.DataFrame,
            user_instruction: str = None,
            func: Callable = None,
            input_column: str = None,
            output_column: str = None,
            strategy: str = "plain_llm",
            *args, 
            **kwargs
    ):  
        if user_instruction is None and func is None:
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return MapOpOutputs(field_name=output_column, output=None)
        
        processed_data = input_data[input_column]
        if isinstance(processed_data.dtype, ImageDtype):
            dtype = "image"
        else:
            dtype = "str"
        if strategy == "plain_llm":
            execution_func = functools.partial(self._plain_llm_execute, dtype=dtype, field_name=input_column, **kwargs)
        elif strategy == "cot":
            demos = kwargs.get("demos", None)
            execution_func = functools.partial(self._llm_cot_execute, dtype=dtype, demos=demos, field_name=input_column, **kwargs)
        else:
            raise NotImplementedError(f"Strategy {strategy} is not implemented.")
        
        map_results, token_cost = [], 0
        for data in processed_data:
            if func is not None:
                try:
                    output = func(data)
                    cost = 0
                except Exception as e:
                    output, cost = execution_func(data, user_instruction)
            else:
                output, cost = execution_func(data, user_instruction)
            output = output if output is not None else "None"
            map_results.append(output)
            token_cost += cost
        
        return MapOpOutputs(
            field_name=output_column,
            output=map_results,
            cost=token_cost
        )
