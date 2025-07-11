import asyncio
import pandas as pd

from nirvana.models.llm_backbone import LLMClient
from nirvana.optim.optimize_prompt import RESULT_EVALUATE_PROMPT


def table_serialize(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data in the output."
    return df.to_json(orient="records", lines=True).strip()


class Evaluator:
    @staticmethod
    def evaluate(ground_truth: pd.DataFrame, process_result: pd.DataFrame, evaluator: LLMClient):
        if not ground_truth.columns.equals(process_result.columns):
            return 0.0
        
        if ground_truth.size != process_result.size:
            return 0.0
        
        evaluate_prompt = [{
            "role": "user",
            "content": RESULT_EVALUATE_PROMPT.format(
                ground_truth=table_serialize(ground_truth),
                result=table_serialize(process_result)
            )
        }]
        rating = asyncio.run(evaluator(messages=evaluate_prompt, parse_tags=True, tags=["score"])["score"])
        rating = float(rating) / 10.0
        return rating
