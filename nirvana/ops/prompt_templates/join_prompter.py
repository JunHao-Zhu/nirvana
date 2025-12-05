from typing import Any


class JoinPrompter:
    def __init__(self):
        self.task_instruction: str = None
        self.output_format: str = None

    def prepare_prompt_for_nested_join(self):
        self.task_instruction = (
            "You are a helpful assistant helping the user make sense of their data. "
            "You are performing a join operation (left data and right data -> True or False) to "
            "determine whether the two given data satisfy the given conditions."
        )
        self.output_format = (
            "Output the result of the join operation concisely in the following format.\n"
            "<output> Your final answer from [True, False] </output>\n"
        )

    def generate_nested_join_prompt(
        self, 
        left_data: Any,
        right_data: Any,
        user_instruction: str | list[str],
        left_dtype: str = "str",
        right_dtype: str = "str",
    ):
        # 1. Prepare system message
        if self.task_instruction is None or self.output_format is None:
            self.prepare_prompt_for_nested_join()
        sys_message = [
            {"role": "system", "content": self.task_instruction},
            {"role": "system", "content": self.output_format}
        ]

        # 2. Prepare data
        user_content = []
        if left_dtype == "str":
            user_content.append({"type": "input_text", "text": f"Left data:\n{left_data}"})
        elif left_dtype == "image":
            user_content.append({"type": "input_text", "text": "Left data:"})
            user_content.append({"type": "input_image", "image_url": left_data})
        else:
            raise ValueError(f"Data type of left data {type(left_data)} is not supported.")

        if right_dtype == "str":
            user_content.append({"type": "input_text", "text": f"Right data:\n{right_data}"})
        elif right_dtype == "image":
            user_content.append({"type": "input_text", "text": "Right data:"})
            user_content.append({"type": "input_image", "image_url": right_data})
        else:
            raise ValueError(f"Data type of right data {type(right_data)} is not supported.")
        
        # 3. Prepare the given condition
        if isinstance(user_instruction, str):
            conditions = f"condition: {user_instruction}"
        elif isinstance(user_instruction, list):
            conditions = [f"condition {idx}: {cond}" for idx, cond in enumerate(user_instruction)]
            conditions = "\n".join(conditions)
        user_content.append({"type": "input_text", "text": conditions})
        
        user_message = [{"role": "user", "content": user_content}]

        messages = sys_message + user_message
        return messages
    
    def prepare_prompt_for_block_join(self):
        self.task_instruction = (
            "Identify pairs of items from the left and right data batches that satisfy the given join conditions."
        )
        self.output_format = (
            "Output only the IDs of pairs satisfying the join conditions in the format of L#-R# (e.g., L3-R5), separated by commas. ",
            "Make sure that all candidate pairs of items are considered.\n",
            "The result of the join operation concisely in the following format.\n",
            "<output> L3-R5,L4-R2,L1-R1 </output>\n"
            "If no pairs satisfy the conditions, output an empty <output></output> tag.\n"
        )
    
    def generate_batch_join_prompt(
        self,
        left_batch: list[Any],
        right_batch: list[Any],
        user_instruction: str | list[str],
        left_dtype: str = "str",
        right_dtype: str = "str",
    ):
        # 1. Prepare system message
        if self.task_instruction is None or self.output_format is None:
            self.prepare_prompt_for_block_join()
        sys_message = [
            {"role": "system", "content": self.task_instruction},
            {"role": "system", "content": self.output_format}
        ]

        # 2. Prepare data
        user_content = []
        # 2.1 Prepare left batch
        user_content.append({"type": "input_text", "text": "Left batch:"})
        if left_dtype == "str":
            for idx, left_data in enumerate(left_batch):
                user_content.append({"type": "input_text", "text": f"L{idx}: {left_data}"})
        elif left_dtype == "image":
            for idx, left_data in enumerate(left_batch):
                user_content.append({"type": "input_text", "text": f"L{idx}:"})
                user_content.append({"type": "input_image", "image_url": left_data})
        else:
            raise ValueError(f"Data type of left data {type(left_data)} is not supported.")
        # 2.2 Prepare right batch
        user_content.append({"type": "input_text", "text": "Right batch:"})
        if right_dtype == "str":
            for idx, right_data in enumerate(right_batch):
                user_content.append({"type": "input_text", "text": f"R{idx}: {right_data}"})
        elif right_dtype == "image":
            for idx, right_data in enumerate(right_batch):
                user_content.append({"type": "input_text", "text": f"R{idx}:"})
                user_content.append({"type": "input_image", "image_url": right_data})
        else:
            raise ValueError(f"Data type of right data {type(right_data)} is not supported.")
        
        # 3. Prepare the given condition
        if isinstance(user_instruction, str):
            conditions = f"condition: {user_instruction}"
        elif isinstance(user_instruction, list):
            conditions = [f"condition {idx}: {cond}" for idx, cond in enumerate(user_instruction)]
            conditions = "\n".join(conditions)
        user_content.append({"type": "input_text", "text": conditions})
        
        user_message = [{"role": "user", "content": user_content}]
        messages = sys_message + user_message
        return messages
