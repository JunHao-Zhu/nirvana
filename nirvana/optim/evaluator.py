import pandas as pd

from nirvana.executors.llm_backbone import LLMClient


def table_serialize(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data in the output."
    return df.to_json(orient="records", lines=True).strip()


class Evaluator:
    evaluate_prompt = """Here are a golden analysis result obtained by a golden data processing plan and a result derived from an alternative data processing plan.
Evaluate the two data analysis results and return a rating between 0 and 10, where 0 means the two results are completely different and 10 means they are exactly the same.

Ground truth:
{ground_truth}

Result from the alternative plan:
{result}

You should carefully consider all values (and their semantics) in both analysis results. The rating score is enclosed within <score></score> tags, i.e., <score>Rating Score</score>.
"""

    @classmethod
    async def evaluate(cls, ground_truth: pd.DataFrame, process_result: pd.DataFrame, evaluator: LLMClient):
        if not ground_truth.columns.equals(process_result.columns):
            return 0.0
        
        if ground_truth.size != process_result.size:
            return 0.0
        
        evaluate_prompt = cls.evaluate_prompt.format(
            ground_truth=table_serialize(ground_truth),
            result=table_serialize(process_result)
        )
        response = await evaluator(messages=evaluate_prompt, parse_tags=True, tags=["score"])
        rating = float(response["score"]) / 10.0
        return rating
