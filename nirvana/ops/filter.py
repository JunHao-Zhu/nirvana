import functools
import asyncio
from typing import Any, Iterable, Callable, Literal
from dataclasses import dataclass
import pandas as pd

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.executors.tools import BaseTool, FunctionCallTool
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.filter_prompter import FilterPrompter


def filter_wrapper(
    input_data: pd.DataFrame, 
    user_instruction: str = None,
    input_columns: list[str] = None,
    func: Callable = None,
    context: list[dict] | str | None = None,
    model: str | None = None,
    strategy: Literal["plain", "fewshot", "self-refine"] = "plain",
    rate_limit: int = 16,
    assertions: list[Callable] | None = [],
    **kwargs
):
    """
    A function wrapper for filter operation

    Args:
        input_data (pd.DataFrame): Input dataframe
        user_instruction (str, optional): User instruction. Defaults to None.
        input_columns (list[str], optional): Input columns. Defaults to None.
        func (Callable, optional): User function. Defaults to None.
        context (list[dict] | str, optional): Context. Defaults to None.
        model (str, optional): Model. Defaults to None.
        strategy (Literal["plain", "fewshot", "self-refine"], optional): Strategy. Defaults to "plain".
        rate_limit (int, optional): Rate limit. Defaults to 16.
        assertions (list[Callable], optional): Assertions. Defaults to [].
        **kwargs: Additional keyword arguments for OpenAI Clent.
    """
    
    filter_op = FilterOperation(
        user_instruction=user_instruction,
        input_columns=input_columns,
        context=context,
        model=model,
        tool=FunctionCallTool.from_function(func=func) if func else None,
        strategy=strategy,
        rate_limit=rate_limit,
        assertions=assertions,
    )
    outputs = asyncio.run(filter_op.execute(
        input_data=input_data,
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
    """
    Filter operator: Uses an LLM to evaluate a natural language predicate on a column
    """
    strategy_options = ["plain", "fewshot", "self_refine"]
    
    def __init__(
        self,
        user_instruction: str = "",
        input_columns: list[str] = [],
        context: list[dict] | str | None = None,
        model: str | None = None,
        tool: BaseTool | None = None,
        strategy: Literal["plain", "fewshot", "self-refine"] = "plain",
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        super().__init__(
            op_name="filter",
            user_instruction=user_instruction,
            context=context,
            model=model,
            tool=tool,
            strategy=strategy,
            rate_limit=rate_limit,
            assertions=assertions,
        )
        self.prompter = FilterPrompter()
        self.input_columns = input_columns

    @property
    def dependencies(self) -> list[str]:
        return self.input_columns
    
    @property
    def generated_fields(self) -> list[str]:
        return []
    
    @property
    def op_kwargs(self) -> dict:
        kwargs = super().op_kwargs
        kwargs["input_columns"] = self.input_columns
        return kwargs

    async def _execute_by_plain_llm(self, data: pd.Series, user_instruction: str, dtypes: list[str], **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_prompt(data, user_instruction, dtypes)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output["output"], output["cost"]

    async def _execute_by_fewshot_llm(self, data: pd.Series, user_instruction: str, dtypes: list[str], demos, **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_fewshot_prompt(data, user_instruction, dtypes, demos)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output["output"], output["cost"]
        
    async def _execute_by_self_refine(self, data: pd.Series, user_instruction: str, dtypes: list[str], **kwargs):
        async with self.semaphore:
            self_refine_cost = 0.0
            generate_prompt = self.prompter.generate_prompt(data, user_instruction, dtypes)
            output = await self.llm(generate_prompt, parse_tags=True, tags=["output"], **kwargs)
            self_refine_cost += output["cost"]

            evaluate_prompt = self.prompter.generate_evaluate_prompt(data, output["raw_output"], user_instruction, dtypes)
            evaluate_output = await self.llm(evaluate_prompt, parse_tags=True, tags=["evaluation", "feedback"], **kwargs)
            self_refine_cost += evaluate_output["cost"]
            
            if "pass" in evaluate_output["evaluation"].lower():
                return output["output"], self_refine_cost
            else:
                refine_prompt = self.prompter.generate_refine_prompt(data, output["raw_output"], user_instruction, evaluate_output["feedback"], dtypes)
                refine_output = await self.llm(refine_prompt, parse_tags=True, tags=["output"], **kwargs)
                self_refine_cost += refine_output["cost"]
                return refine_output["output"], self_refine_cost
    
    async def _execute_by_func(self, data: pd.Series, user_instruction: str, func: Callable, llm_call: Callable, **kwargs):
        try:
            output = func(data)
            return output, 0.0
        except Exception as e:
            return await llm_call(data, user_instruction)
    
    def _postprocess_filter_outputs(self, results: Iterable[tuple[Any, float]]):
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
        **kwargs
    ):
        if self.user_instruction is None and not self.has_udf():
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return FilterOpOutputs()
        
        processed_data = input_data[self.input_columns]
        dtypes = []
        for col in self.input_columns:
            if isinstance(input_data[col].dtype, ImageDtype):
                dtypes.append("image")
            else:
                dtypes.append("str")

        if self.strategy == "plain":
            execution_func = functools.partial(self._execute_by_plain_llm, dtypes=dtypes, field_name=self.input_columns[0], model=self.model, **kwargs)
        elif self.strategy == "fewshot":
            assert self.context is not None, "Few-shot examples must be provided in the context for in-context learning."
            demos = self.context
            execution_func = functools.partial(self._execute_by_fewshot_llm, dtypes=dtypes, demos=demos, field_name=self.input_columns[0], model=self.model, **kwargs)
        elif self.strategy == "self_refine":
            execution_func = functools.partial(self._execute_by_self_refine, dtypes=dtypes, field_name=self.input_columns[0], model=self.model, **kwargs)
        else:
            raise ValueError(f"The optional strategies available for filter are {self.strategy_options}. Strategy {self.strategy} is not supported.")

        # Create tasks for all data points
        tasks = []
        for _, data in processed_data.iterrows():
            if data.hasnans:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=(False, 0.0))))
            elif self.has_udf():
                tasks.append(asyncio.create_task(self._execute_by_func(data, self.user_instruction, self.tool, execution_func)))
            else:
                tasks.append(asyncio.create_task(execution_func(data, self.user_instruction)))

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Process results
        filter_outputs, token_cost = self._postprocess_filter_outputs(results)
        return FilterOpOutputs(
            output=filter_outputs,
            cost=token_cost
        )
