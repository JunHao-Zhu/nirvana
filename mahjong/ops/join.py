"""
Join: Join values of two columns against a specific user's instruction.
"""
import pandas as pd
from typing import Any, List, Iterable
from dataclasses import dataclass

from mahjong.ops.base import BaseOpOutputs, BaseOperation
from mahjong.prompt_templates.join_prompter import JoinPrompter


def join_wrapper(
    left_data: pd.DataFrame,
    right_data: pd.DataFrame, 
    user_instruction: str,
    left_column: str,
    right_column: str, 
    **kwargs
):
    join_op = JoinOperation()
    outputs = join_op.execute(
        left_data=left_data,
        right_data=right_data,
        user_instruction=user_instruction,
        left_column=left_column,
        right_column=right_column,
        strategy=None,
        **kwargs
    )
    return outputs


@dataclass
class JoinOpOutputs(BaseOpOutputs):
    output: Iterable[bool] = None
    joined_pairs: List[tuple] = None


class JoinOperation(BaseOperation):
    """ TODO: Implement JoinOperation class """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("join", *args, **kwargs)
        self.prompter = JoinPrompter()

    def _plain_llm_execute(self, left_data: Any, right_data: Any, user_instruction: str):
        outputs = []
        total_cost = 0
        for left_value, right_value in zip(left_data, right_data):
            full_prompt = self.prompter.generate_prompt(left_value, right_value, user_instruction)
            output = self.llm(full_prompt, parse_tags=True, tags=["output"])
            outputs.append(output["output"])
            total_cost += output["cost"]
        return outputs, total_cost
    
    def _postprocess_llm_outputs(self, data_id_pairs: List[tuple], llm_outputs: List[Any]):
        outputs = []
        for output in llm_outputs:
            if "True" in output:
                outputs.append(True)
            elif "False" in output:
                outputs.append(False)
            else:
                raise ValueError("The llm outputs do not contain True or False.")
            
        joined_pairs = []
        for i, output in enumerate(outputs):
            if output:
                joined_pairs.append(data_id_pairs[i])
        return joined_pairs, outputs

    def execute(
            self, 
            left_data: pd.DataFrame,
            right_data: pd.DataFrame,
            user_instruction: str,
            left_column: str,
            right_column: str,
            strategy: str = None,
            *args, 
            **kwargs
    ):
        left_ids = left_data[left_column].index
        right_ids = right_data[right_column].index
        data_id_pairs = [(left_id, right_id) for left_id in left_ids for right_id in right_ids]

        outputs, cost = self._plain_llm_execute(left_data, right_data, user_instruction)
        joined_pairs, outputs = self._postprocess_llm_outputs(data_id_pairs, outputs)
        return JoinOpOutputs(
            output=outputs,
            joined_pairs=joined_pairs,
            cost=cost,
        )
