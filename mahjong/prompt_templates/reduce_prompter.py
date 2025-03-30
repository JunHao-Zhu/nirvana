import mahjong as mjg
from typing import Any, Iterable


class ReducePrompter:
    """
    TODO: Simple implementation that does not consider the case where the input length exceeds the token limit.
    The next step is to implement several optimizations, like `summarize and aggregate` and `incremental aggregation`
    """
    
    def __init__(self):
        self.system_instruction = (
            "You are a helpful assistant helping the user make sense of their data. "
            "You are performing a reduce operation (multiple inputs: one output) to "
            "aggregate the given data based on the user's instruction.\n"
            "Output the result of the reduce operation concisely in the following format.\n"
            "<output> The aggregation result of the given data </output>\n"
        )

    def generate_prompt(
            self, 
            data_set: Iterable[Any],
            user_instruction: str,
    ):
        # 1. Prepare system message
        sys_message = [{"role": "system", "content": self.system_instruction}]

        # 2. Prepare user message
        for ctr, data in enumerate(data_set):
            if isinstance(data, str):
                user_content = [{"type": "text", "text": f"Data {ctr}: {data}"}]
            elif isinstance(data, mjg.ImageDtype):
                user_content = [
                    {"type": "text", "text": f"Data {ctr}: "},
                    {"type": "image", "image_url": {"url": data}}
                ]
            else:
                raise ValueError(f"Data type {type(data)} is not supported.")
        user_content.append({"type": "text", "text": user_instruction})
        user_message = [{"role": "user", "content": user_content}]

        messages = sys_message + user_message
        return messages
