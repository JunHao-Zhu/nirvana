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
            "<output> LLM output </output>\n"
        )

    def generate_prompt(
            self, 
            data_set: Iterable[Any],
            user_instruction: str,
            dtype: str = "str",
    ):
        # 1. Prepare system message
        sys_message = [{"role": "system", "content": self.system_instruction}]

        # 2. Prepare user message
        for ctr, data in enumerate(data_set):
            if dtype == "str":
                user_content = [{"type": "text", "text": f"Data {ctr}: {str(data)}"}]
            elif dtype == "image":
                user_content = [
                    {"type": "text", "text": f"Data {ctr}: "},
                    {"type": "image", "image_url": {"url": data}}
                ]
            else:
                raise ValueError(f"Data type {dtype} is not supported.")
        user_content.append({"type": "text", "text": user_instruction})
        user_message = [{"role": "user", "content": user_content}]

        messages = sys_message + user_message
        return messages
