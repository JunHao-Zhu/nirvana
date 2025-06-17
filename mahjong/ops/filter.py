"""
Filter: remove data that violates NL predicates.
"""
import functools
import asyncio
from typing import Any, Union, Iterable, Callable, Tuple
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
    outputs = asyncio.run(filter_op.execute(
        input_data=input_data,
        user_instruction=user_instruction,
        func=func,
        input_column=input_column,
        strategy=strategy,
        **kwargs
    ))
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

    async def _execute_by_plain_llm(self, data: Any, user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            if dtype == "str":
                data = f"{kwargs['field_name']}: {str(data)}"
            full_prompt = self.prompter.generate_prompt(data, user_instruction, dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output["output"], output["cost"]

    async def _execute_by_fewshot_llm(self, data: Any, user_instruction: str, dtype: str, demos, **kwargs):
        async with self.semaphore:
            if dtype == "str":
                data = f"{kwargs['field_name']}: {str(data)}"
            full_prompt = self.prompter.generate_fewshot_prompt(data, user_instruction, dtype, demos)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output["output"], output["cost"]
    
    async def _execute_by_func(self, data: Any, user_instruction: str, func: Callable, llm_call: Callable, **kwargs):
        try:
            output = func(data)
            return output, 0.0
        except Exception as e:
            return await llm_call(data, user_instruction)
    
    def _postprocess_filter_outputs(self, results: Iterable[Tuple[Any, float]]):
        outputs, costs = [], 0.0
        for output, cost in results:
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
            costs += cost
        return outputs, costs

    async def execute(
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
            execution_func = functools.partial(self._execute_by_plain_llm, dtype=dtype, field_name=input_column, **kwargs)
        elif strategy == "fewshot":
            demos = kwargs.get("demos", None)
            execution_func = functools.partial(self._execute_by_fewshot_llm, dtype=dtype, demos=demos, field_name=input_column, **kwargs)
        else:
            raise NotImplementedError(f"Strategy {strategy} is not implemented.")
        
        # Create tasks for all data points
        tasks = []
        for data in processed_data:
            if pd.isna(data):
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=(False, 0.0))))
            elif func is not None:
                tasks.append(asyncio.create_task(self._execute_by_func(data, user_instruction, func, execution_func)))
            else:
                tasks.append(asyncio.create_task(execution_func(data, user_instruction)))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Process results
        filter_outputs, token_cost = self._postprocess_filter_outputs(results)
        return FilterOpOutputs(
            output=filter_outputs,
            cost=token_cost
        )
