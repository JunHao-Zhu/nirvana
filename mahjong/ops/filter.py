"""
Filter: remove data that violates NL predicates.
"""
import functools
from typing import Any, Iterable
from dataclasses import dataclass
import pandas as pd

from mahjong.ops.base import BaseOpOutputs, BaseOperation
from mahjong.prompt_templates.filter_prompter import FilterPrompter


def filter_wrapper(
    input_data: pd.DataFrame, 
    user_instruction: str, 
    input_column: str,
    strategy: str = None, 
    **kwargs
):
    filter_op = FilterOperation()
    outputs = filter_op.execute(
        input_data=input_data,
        user_instruction=user_instruction,
        input_column=input_column,
        strategy=strategy,
        **kwargs
    )
    return outputs


@dataclass
class FilterOpOutputs(BaseOpOutputs):
    output: Iterable[bool] = None


class FilterOperation(BaseOperation):
    """ TODO: Implement FilterOperation class """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("filter", *args, **kwargs)
        self.prompter = FilterPrompter()
    
    def _plain_llm_execute(self, processed_data: Iterable[Any], user_instruction: str):
        outputs = []
        total_cost = 0
        for data in processed_data:
            full_prompt = self.prompter.generate_prompt(user_instruction, data)
            output = self.llm(full_prompt, parse_tags=True, tags=["output"])
            outputs.append(output["output"])
            total_cost += output["cost"]
        return outputs, total_cost

    def _llm_cot_execute(self, processed_data: Iterable[Any], user_instruction: str, demos):
        outputs = []
        total_cost = 0
        for data in processed_data:
            full_prompt = self.prompter.generate_cot_prompt(user_instruction, data, demos)
            output = self.llm(full_prompt, parse_tags=True, tags=["output"])
            outputs.append(output["output"])
            total_cost += output["cost"]
        return outputs, total_cost
    
    def _postprocess_llm_outputs(self, llm_outputs: Iterable[str]):
        outputs = []
        for output in llm_outputs:
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
            user_instruction: str,
            input_column: str,
            strategy: str = "plain_llm",
            *args, 
            **kwargs
    ):
        if input_data.empty:
            return FilterOpOutputs()
        
        if strategy == "plain_llm":
            execution_func = functools.partial(self._plain_llm_execute)
        elif strategy == "cot":
            demos = kwargs.get("demos", None)
            execution_func = functools.partial(self._llm_cot_execute, demos=demos)
        else:
            raise NotImplementedError(f"Strategy {strategy} is not implemented.")
        
        processed_data = input_data[input_column]
        outputs, cost = execution_func(processed_data, user_instruction)
        
        outputs = self._postprocess_llm_outputs(outputs)
        return FilterOpOutputs(
            output=outputs,
            cost=cost
        )
