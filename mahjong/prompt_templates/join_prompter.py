import mahjong as mjg
from typing import Any, List, Dict, Union


class JoinPrompter:
    def __init__(self):
        self.system_instruction = (
            "You are a helpful assistant helping the user make sense of their data. "
            "You are performing a join operation (left data and right data -> True or False) to "
            "determine whether the left data and right data satisfy the given conditions. "
            "The answer should strictly be either True or False.\n"
            "Output the result of the join operation concisely in the following format.\n"
            "<output> Your final answer from [True, False] </output>\n"
        )

    def generate_prompt(
            self, 
            left_data: Any,
            right_data: Any,
            user_instruction: Union[str, List[str]],
    ):
        # 1. Prepare system message
        sys_message = [{"role": "system", "content": self.system_instruction}]

        # 2. Prepare data
        user_content = []
        if isinstance(left_data, str):
            user_content.append({"type": "text", "text": f"Left data:\n{left_data}"})
        elif isinstance(left_data, mjg.ImageDtype):
            user_content.append({"type": "text", "text": "Left data:"})
            user_content.append({"type": "image", "image_url": {"url": left_data}})
        else:
            raise ValueError(f"Data type of left data {type(left_data)} is not supported.")

        if isinstance(right_data, str):
            user_content.append({"type": "text", "text": f"Right data:\n{right_data}"})
        elif isinstance(right_data, mjg.ImageDtype):
            user_content.append({"type": "text", "text": "Right data:"})
            user_content.append({"type": "image", "image_url": {"url": right_data}})
        else:
            raise ValueError(f"Data type of right data {type(right_data)} is not supported.")
        
        # 3. Prepare the given condition
        if isinstance(user_instruction, str):
            conditions = f"condition: {user_instruction}"
        elif isinstance(user_instruction, list):
            conditions = [f"condition {idx}: {cond}" for idx, cond in enumerate(user_instruction)]
            conditions = "\n".join(conditions)
        user_content.append({"type": "text", "text": conditions})
        
        user_message = [{"role": "user", "content": user_content}]

        messages = sys_message + user_message
        return messages
