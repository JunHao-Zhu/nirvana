from typing import Any, Iterable


class RankPrompter:
    
    def __init__(self):
        self.system_instruction = (
            "Your task is to select the most relevant item from the given two data to the user-specified criterion. "
            "Responde only with the label of the data such as 'item number', i.e., "
            "the response must be either 1 or 2 (things like 'None' or 'Neither' are forbidden), "
            "depending on which item is more relevant to the criterion.\n"
            "Output the item number concisely in the following format.\n"
            "<output> item number (1 or 2) </output>\n"
        )

    def generate_prompt(
            self, 
            data1: Any,
            data2: Any,
            user_instruction: str,
            dtype: str = "str",
    ):
        # 1. Prepare system message
        sys_message = [{"role": "system", "content": self.system_instruction}]

        # 2. Prepare user message
        user_content = []
        if dtype == "str":
            user_content.append({"type": "input_text", "text": f"Data item 1: {str(data1)}"})
            user_content.append({"type": "input_text", "text": f"Data item 2: {str(data2)}"})
        elif dtype == "image":
            user_content.append({"type": "input_text", "text": f"Data item 1: "})
            user_content.append({"type": "input_image", "image_url": data1})
            user_content.append({"type": "input_text", "text": f"Data item 2: "})
            user_content.append({"type": "input_image", "image_url": data2})
        else:
            raise ValueError(f"Data type {dtype} is not supported.")
        user_content.append({"type": "input_text", "text": f"Criterion: {user_instruction}"})
        user_message = [{"role": "user", "content": user_content}]

        messages = sys_message + user_message
        return messages
