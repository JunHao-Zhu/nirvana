from typing import Any, List, Dict, Union


class FilterPrompter:
    def __init__(self):
        self.system_instruction = (
            "You are a helpful assistant helping the user make sense of their data. "
            "You are performing a filter operation (one input: True or False) to "
            "determine whether the given data satisfies the given condition(s). "
            "The answer should strictly be either True or False.\n"
            "Output the result of the filter operation concisely in the following format.\n"
            "<output> True or False </output>\n"
        )

    def generate_prompt(
            self,
            data: Any,
            user_instruction: str,
            dtype: str = "str"
    ):
        # 1. Prepare system message
        sys_message = [{"role": "system", "content": self.system_instruction}]

        # 2. Prepare data
        user_content = []
        if dtype == "str":
            user_content.append({"type": "text", "text": data})
        elif dtype == "image":
            user_content.append(
                {"type": "image_url", "image_url": {"url": data}}
            )
        else:
            raise ValueError(f"Data type {dtype} is not supported.")
        
        # 3. Prepare the given condition
        conditions = f"condition: {user_instruction}"
        user_content.append({"type": "text", "text": conditions})
        
        user_message = [{"role": "user", "content": user_content}]

        messages = sys_message + user_message
        return messages
    
    def generate_cot_prompt(
            self,
            data: Any,
            user_instruction: str,
            dtype: str,
            demos: List[Dict[str, Any]]
    ):
        # 1. Prepare system message
        sys_message = [{"role": "system", "content": self.system_instruction}]

        # 2. Prepare demonstration message
        demos_message = []
        for demo in demos:
            demo_content = []
            if dtype == str:
                demo_content = [
                    {"type": "text", "text": demo["data"]},
                    {"type": "text", "text": user_instruction},
                    {"type": "text", "text": demo["answer"]}
                ]
            elif dtype == "image":
                demo_content = [
                    {"type": "image_url", "image_url": {"url": demo["data"]}},
                    {"type": "text", "text": user_instruction},
                    {"type": "text", "text": demo["answer"]}
                ]
            else:
                raise ValueError(f"Data type {dtype} is not supported.")
            demos_message.append(
                {"role": "assistant", "content": demo_content}
            )

        # 3. Prepare user message
        if dtype == "str":
            user_content = [
                {"type": "text", "text": data},
                {"type": "text", "text": user_instruction}
            ]
        elif dtype == "image":
            user_content = [
                {"type": "image_url", "image_url": {"url": data}},
                {"type": "text", "text": user_instruction}
            ]
        else:
            raise ValueError(f"Data type {dtype} is not supported.")
        user_message = [{"role": "user", "content": user_content}]
        
        messages = sys_message + demos_message + user_message
        return messages
