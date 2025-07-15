import functools
import asyncio
from typing import Any, Iterable, Callable, Tuple
from dataclasses import dataclass
import pandas as pd

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.map_prompter import MapPrompter


def map_wrapper(
    input_data: pd.DataFrame, 
    user_instruction: str = None,
    func: Callable = None, 
    input_column: str = None,
    output_column: str = None, 
    strategy: str = None,
    **kwargs
):
    map_op = MapOperation()
    outputs = asyncio.run(map_op.execute(
        input_data=input_data,
        user_instruction=user_instruction,
        input_column=input_column,
        output_column=output_column,
        strategy=strategy,
        **kwargs
    ))
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
    """
    Map operator: Applies an LLM to perform a transformation on a column, producing a new column as output
    """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("map", *args, **kwargs)
        self.prompter = MapPrompter()
        rate_limit = kwargs.get("rate_limit", 16)
        self.semaphore = asyncio.Semaphore(rate_limit)
    
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
    
    def _postprocess_map_outputs(self, results: Iterable[Tuple[Any, float]]):
        outputs, costs = [], 0.0
        for output, cost in results:
            output = output if output is not None else "None"
            outputs.append(output)
            costs += cost
        return outputs, costs

    async def execute(
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
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=("None", 0.0))))
            elif func is not None:
                tasks.append(asyncio.create_task(self._execute_by_func(data, user_instruction, func, execution_func)))
            else:
                tasks.append(asyncio.create_task(execution_func(data, user_instruction)))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Process results
        map_results, token_cost = self._postprocess_map_outputs(results)
        return MapOpOutputs(
            field_name=output_column,
            output=map_results,
            cost=token_cost
        )
