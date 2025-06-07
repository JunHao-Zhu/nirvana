"""
Filter: remove data that violates NL predicates.
"""
import functools
from typing import Any, Union, Iterable, Callable
from dataclasses import dataclass
import pandas as pd

from mahjong.dataframe.arrays.image import ImageDtype
from mahjong.ops.base import BaseOpOutputs, BaseOperation
from mahjong.ops.prompt_templates.filter_prompter import FilterPrompter


def filter_wrapper(
    input_data: pd.DataFrame, 
    user_instruction: str = None,
    func: Callable = None, 
    input_column: str = None,
    strategy: str = None, 
    **kwargs
):
    filter_op = FilterOperation()
    outputs = filter_op.execute(
        input_data=input_data,
        user_instruction=user_instruction,
        func=func,
        input_column=input_column,
        strategy=strategy,
        **kwargs
    )
    return outputs


@dataclass
class FilterOpOutputs(BaseOpOutputs):
    output: Iterable[bool] = None

    def __add__(self, other: "FilterOpOutputs"):
        return FilterOpOutputs(
            output=self.output + other.output,
            cost=self.cost + other.cost
        )


class FilterOperation(BaseOperation):
    """ TODO: Implement FilterOperation class """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("filter", *args, **kwargs)
        self.prompter = FilterPrompter()
    
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
    
    def _postprocess_filter_outputs(self, llm_outputs: Iterable[Union[str, bool]]):
        outputs = []
        for output in llm_outputs:
            if output is None:
                outputs.append(False)
                continue
            if isinstance(output, bool):
                outputs.append(output)
                continue
            if "True" in output:
                outputs.append(True)
            elif "False" in output:
                outputs.append(False)
            else:
                raise ValueError("The llm outputs do not contain True or False.")
        return outputs

    def execute(
            self, 
            input_data: pd.DataFrame,
            user_instruction: str = None,
            func: Callable = None,
            input_column: str = None,
            strategy: str = "plain_llm",
            *args, 
            **kwargs
    ):
        if user_instruction is None and func is None:
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return FilterOpOutputs()
        
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
        
        filter_outputs, token_cost = [], 0
        for data in processed_data:
            if pd.isna(data):
                output, cost = False, 0.0
            elif func is not None:
                try:
                    output = func(data)
                    cost = 0
                except Exception as e:
                    output, cost = execution_func(data, user_instruction)
            else:
                output, cost = execution_func(data, user_instruction)
            filter_outputs.append(output)
            token_cost += cost
        
        filter_outputs = self._postprocess_filter_outputs(filter_outputs)
        return FilterOpOutputs(
            output=filter_outputs,
            cost=token_cost
        )
