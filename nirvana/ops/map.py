import warnings

import functools
import asyncio
from typing import Any, Iterable, Callable, Literal
from dataclasses import dataclass, field
from numpy import amax
import pandas as pd

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.executors.tools import BaseTool, FunctionCallTool
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.map_prompter import MapPrompter


def map_wrapper(
    input_data: pd.DataFrame, 
    user_instruction: str = None,
    input_columns: list[str] = None,
    output_columns: list[str] = None,
    context: list[dict] | str | None = None,
    model: str | None = None,
    func: Callable = None,
    strategy: Literal["plain", "fewshot", "self-refine"] = "plain",
    limit: int | None = None,
    rate_limit: int = 16,
    assertions: list[Callable] | None = [],
    **kwargs
):
    """
    A function wrapper for map operation

    Args:
        input_data (pd.DataFrame): Input dataframe
        user_instruction (str, optional): User instruction. Defaults to None.
        input_columns (list[str], optional): Input column. Defaults to None.
        output_columns (list[str], optional): Output columns. Defaults to None.
        context (list[dict] | str, optional): Context. Defaults to None.
        model (str, optional): Model. Defaults to None.
        func (Callable, optional): User function. Defaults to None.
        strategy (Literal["plain", "fewshot", "self-refine"], optional): Strategy. Defaults to "plain".
        limit (int): Maximum number of outputs to produce before stopping.
        rate_limit (int, optional): Rate limit. Defaults to 16.
        assertions (list[Callable], optional): Assertions. Defaults to [].
        **kwargs: Additional keyword arguments for OpenAI Clent.
    """
    
    map_op = MapOperation(
        user_instruction=user_instruction,
        strategy=strategy,
        input_columns=input_columns,
        output_columns=output_columns,
        context=context,
        model=model,
        tool=FunctionCallTool.from_function(func=func) if func else None,
        limit=limit,
        rate_limit=rate_limit,
        assertions=assertions
    )
    outputs = asyncio.run(map_op.execute(
        input_data=input_data,
        **kwargs
    ))
    return outputs


@dataclass
class MapOpOutputs(BaseOpOutputs):
    field_name: list[str] | None = field(default=None)
    output: dict[str, Iterable] = field(default_factory=list)

    def __add__(self, other: "MapOpOutputs"):
        assert self.field_name == other.field_name, "Cannot merge MapOpOutputs with different field names."
        map_output = dict()
        for name in self.field_name:
            map_output[name] = self.output[name] + other.output[name]
        return MapOpOutputs(
            field_name=self.field_name,
            output=map_output,
            cost=self.cost + other.cost
        )


class MapOperation(BaseOperation):
    """
    Map operator: Applies an LLM to perform a transformation on a column, producing a new column as output
    """
    strategy_options = ["plain", "fewshot", "self-refine"]
    
    def __init__(
        self,
        user_instruction: str = "",
        input_columns: list[str] = [],
        output_columns: list[str] = [],
        context: list[dict] | str | None = None,
        model: str | None = None,
        tool: Callable | BaseTool | None = None,
        strategy: Literal["plain", "fewshot", "self-refine"] = "plain",
        limit: int | None = None,
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        if tool and not isinstance(tool, BaseTool):
            tool = FunctionCallTool.from_function(func=tool)
        
        super().__init__(
            op_name="map",
            user_instruction=user_instruction,
            context=context,
            model=model,
            tool=tool,
            strategy=strategy,
            limit=limit,
            rate_limit=rate_limit,
            assertions=assertions,
        )
        self.prompter = MapPrompter()
        self.input_columns = input_columns
        self.output_columns = output_columns

    @property
    def dependencies(self) -> list[str]:
        return self.input_columns
    
    @property
    def generated_fields(self) -> list[str]:
        return self.output_columns
    
    @property
    def op_kwargs(self):
        kwargs = super().op_kwargs
        kwargs["input_columns"] = self.input_columns
        kwargs["output_columns"] = self.output_columns
        return kwargs

    async def _execute_by_plain_llm(self, data: pd.Series, user_instruction: str, dtypes: list[str], **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_prompt(data, user_instruction, self.output_columns, dtypes)
            output = await self.llm(full_prompt, parse_tags=True, tags=self.output_columns, **kwargs)
            result = self._postprocess_map_output(output, self.output_columns)
            return result, output["cost"]

    async def _execute_by_fewshot_llm(self, data: pd.Series, user_instruction: str, dtypes: list[str], demos, **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_fewshot_prompt(data, user_instruction, self.output_columns, dtypes, demos)
            output = await self.llm(full_prompt, parse_tags=True, tags=self.output_columns, **kwargs)
            result = self._postprocess_map_output(output, self.output_columns)
            return result, output["cost"]
        
    async def _execute_by_self_refine(self, data: pd.Series, user_instruction: str, dtypes: list[str], **kwargs):
        async with self.semaphore:
            self_refine_cost = 0.0
            generate_prompt = self.prompter.generate_prompt(data, user_instruction, self.output_columns, dtypes)
            output = await self.llm(generate_prompt, parse_tags=True, tags=self.output_columns, **kwargs)
            self_refine_cost += output["cost"]

            evaluate_prompt = self.prompter.generate_evaluate_prompt(data, output["raw_output"], user_instruction, dtypes)
            evaluate_output = await self.llm(evaluate_prompt, parse_tags=True, tags=["evaluation", "feedback"], **kwargs)
            self_refine_cost += evaluate_output["cost"]
            
            if "pass" in evaluate_output["evaluation"].lower():
                result = self._postprocess_map_output(output, self.output_columns)
                return result, self_refine_cost
            else:
                refine_prompt = self.prompter.generate_refine_prompt(data, output["raw_output"], user_instruction, self.output_columns, evaluate_output["feedback"], dtypes)
                refine_output = await self.llm(refine_prompt, parse_tags=True, tags=self.output_columns, **kwargs)
                self_refine_cost += refine_output["cost"]
                result = self._postprocess_map_output(refine_output, self.output_columns)
                refine_output["cost"] = self_refine_cost
                return result, self_refine_cost

    async def _execute_by_func(self, data: pd.Series, user_instruction: str, func: Callable, llm_call: Callable, **kwargs):
        try:
            map_result = func(data)
            result = self._postprocess_map_output(map_result, self.output_columns)
            return result, 0.0
        except Exception as e:
            return await llm_call(data, user_instruction)
        
    def _postprocess_map_output(self, llm_result: dict[str, str] | None, output_columns: list[str]) -> dict[str, Any] | None:
        try:
            if llm_result is None:
                return {column: None for column in output_columns}
            else:
                map_output = {column: llm_result.get(column, None) for column in output_columns}
                return map_output
        except KeyError as e:
            raise KeyError(f"Expected field {e} not found in the LLM response.")

    async def execute(
        self, 
        input_data: pd.DataFrame,
        **kwargs
    ):  
        if self.user_instruction is None and not self.has_udf():
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return MapOpOutputs(field_name=self.output_columns, output=[])

        processed_data = input_data[self.input_columns]
        dtypes = []
        for col in self.input_columns:
            if isinstance(input_data[col].dtype, ImageDtype):
                dtypes.append("image")
            else:
                dtypes.append("text")
        
        if self.strategy == "plain":
            execution_func = functools.partial(self._execute_by_plain_llm, dtypes=dtypes, model=self.model, **kwargs)
        elif self.strategy == "fewshot":
            assert self.context is not None, "Few-shot examples must be provided in the context for in-context learning."
            demos = self.context
            execution_func = functools.partial(self._execute_by_fewshot_llm, dtypes=dtypes, demos=demos, model=self.model, **kwargs)
        elif self.strategy == "self_refine":
            execution_func = functools.partial(self._execute_by_self_refine, dtypes=dtypes, model=self.model, **kwargs)
        else:
            raise ValueError(f"The optional strategies available for map are {self.strategy_options}. Strategy {self.strategy} is not supported.")

        # Create tasks for all data points
        tasks = []
        empty_map_result = {column: None for column in self.output_columns}
        for _, data in processed_data.iterrows():
            if data.empty:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=(empty_map_result, 0.0))))
            elif self.has_udf():
                tasks.append(asyncio.create_task(self._execute_by_func(data, self.user_instruction, self.tool, execution_func)))
            else:
                tasks.append(asyncio.create_task(execution_func(data, self.user_instruction)))
        
        # Wait for all tasks to complete
        if self.limit is not None and self.limit <= 0:
            warnings.warn("The limit should be positive. To execute, the limit will be ignored.")
            self.limit = None
        
        token_cost: float = 0.0
        map_outputs: dict[str, list] = {column: [] for column in self.output_columns}
        if self.limit is not None:
            num_passed_records: int = 0
            reach_limit: bool = False
            for i in range(0, len(tasks), self.limit):
                if reach_limit:
                    break
                batch_tasks = tasks[i:i + self.limit]
                batch_results = await asyncio.gather(*batch_tasks)
                token_cost += sum([result[1] for result in batch_results])
                for result, _ in batch_results:
                    num_passed_records += 1
                    for column in self.output_columns:
                        map_outputs[column].append(result[column])
                    if num_passed_records >= self.limit:
                        reach_limit = True
                        break
            num_remaining_records = len(processed_data) - num_passed_records
            if num_remaining_records > 0:
                for column in self.output_columns:
                    map_outputs[column].extend([None] * num_remaining_records)
        else:
            results = await asyncio.gather(*tasks)
            token_cost += sum([result[1] for result in results])
            for result, _ in results:
                for column in self.output_columns:
                    map_outputs[column].append(result[column])
        
        return MapOpOutputs(
            field_name=self.output_columns,
            output=map_outputs,
            cost=token_cost
        )
