from typing import Any
import pandas as pd


class FilterPrompter:
    def __init__(self):
        self.system_instruction = (
            "You are a helpful assistant helping the user make sense of their data. "
            "You are performing a filter operation (data -> True or False) to "
            "determine whether the given data satisfies the given condition(s)."
        )
        self.output_format = (
            "The answer should strictly be either True or False. "
            "Output the result of the filter operation concisely in the following format.\n"
            "<output> True or False </output>\n"
        )

    def generate_prompt(
        self,
        data: pd.Series,
        user_instruction: str,
        dtypes: list[str]
    ):
        # 1. Prepare system message
        sys_message = [
            {"role": "system", "content": self.system_instruction},
            {"role": "system", "content": self.output_format}
        ]

        # 2. Prepare data
        user_content = []
        for dtype, (key, val) in zip(dtypes, data.items()):
            if dtype == "str":
                user_content.append({"type": "input_text", "text": f"{key}: {val}"})
            elif dtype == "image":
                user_content.append({"type": "input_text", "text": f"{key}:"})
                user_content.append({"type": "input_image", "image_url": val})
            elif dtype == "audio":
                user_content.append({"type": "input_text", "text": f"{key}:"})
                user_content.append(
                    {"type": "input_audio", "input_audio": {"data": val, "format": "wav"}}
                )
            else:
                raise ValueError(f"Data type {dtype} is not supported.")
        
        # 3. Prepare the given condition
        conditions = f"condition: {user_instruction}"
        user_content.append({"type": "input_text", "text": conditions})
        
        user_message = [{"role": "user", "content": user_content}]

        messages = sys_message + user_message
        return messages
    
    def generate_fewshot_prompt(
            self,
            data: pd.Series,
            user_instruction: str,
            dtypes: list[str],
            demos: list[dict[str, Any]]
    ):
        # 1. Prepare system message
        sys_message = [
            {"role": "system", "content": self.system_instruction},
            {"role": "system", "content": self.output_format}
        ]

        # 2. Prepare demonstration message
        demos_message = []
        for demo in demos:
            demo_content = []
            demo_data: pd.Series | dict = demo["data"]
            demo_answer: str = demo["answer"]
            for dtype, (key, val) in zip(dtypes, demo_data.items()):
                if dtype == "str":
                    demo_content.append({"type": "input_text", "text": f"{key}: {val}"})
                elif dtype == "image":
                    demo_content.append({"type": "input_text", "text": f"{key}:"})
                    demo_content.append({"type": "input_image", "image_url": val})
                elif dtype == "audio":
                    demo_content.append({"type": "input_text", "text": f"{key}:"})
                    demo_content.append(
                        {"type": "input_audio", "input_audio": {"data": val, "format": "wav"}}
                    )
                else:
                    raise ValueError(f"Data type {dtype} is not supported.")
            demo_content.append({"type": "input_text", "text": f"condition: {user_instruction}"})
            demo_content.append({"type": "input_text", "text": f"Answer: {demo_answer}"})
            demos_message.append(
                {"role": "assistant", "content": demo_content}
            )

        # 3. Prepare user message
        user_content = []
        for dtype, (key, val) in zip(dtypes, data.items()):
            if dtype == "str":
                user_content.append({"type": "input_text", "text": f"{key}: {val}"})
            elif dtype == "image":
                user_content.append({"type": "input_text", "text": f"{key}:"})
                user_content.append({"type": "input_image", "image_url": val})
            elif dtype == "audio":
                user_content.append({"type": "input_text", "text": f"{key}:"})
                user_content.append(
                    {"type": "input_audio", "input_audio": {"data": val, "format": "wav"}}
                )
            else:
                raise ValueError(f"Data type {dtype} is not supported.")
        user_message = [{"role": "user", "content": user_content}]
        
        messages = sys_message + demos_message + user_message
        return messages
    
    def generate_evaluate_prompt(
            self,
            data: pd.Series,
            answer: Any,
            user_instruction: str,
            dtypes: list[str]
    ):
        evaluator_task = [{"role": "user", "content": "Analyze the filter operation tasked with evaluating the condition on the given data:"}]

        _input = [{"type": "input_text", "text": "Input:"}]
        for dtype, (key, val) in zip(dtypes, data.items()):
            if dtype == "str":
                _input.append({"type": "input_text", "text": f"{key}: {val}"})
            elif dtype == "image":
                _input.append({"type": "input_text", "text": f"{key}:"})
                _input.append({"type": "input_image", "image_url": val})
            elif dtype == "audio":
                _input.append({"type": "input_text", "text": f"{key}:"})
                _input.append(
                    {"type": "input_audio", "input_audio": {"data": val, "format": "wav"}}
                )
            else:
                raise ValueError(f"Data type {dtype} is not supported.")
        
        _output = [{"type": "input_text", "text": f"Output:\n{answer}"}]

        instruction = user_instruction.strip() + " " + self.output_format
        _instruction = [{"type": "input_text", "text": f"Instruction:\n{instruction}"}]
        
        evaluator_context = [
            {"role": "user", "content": _input + _output + _instruction}
        ]

        evaluator_criteria = (
            "Evaluate the filter operation based on the following criteria:\n"
            "1. Is the final determination either True or False?\n"
            "2. Does the output strictly adhere to the required output format?\n"
            "3. Whether correctly derives the final True/False determination based on the given instruction.\n"
            "You should be evaluating only and not attemping to solve the task. Only output PASS if all criteria are met and you have no further suggestions for improvements.\n"
            "Output your evaluation concisely in the following format.\n"
            "<evaluation> PASS or FAIL </evaluation>\n"
            "<feedback> What needs improvement and why. </feedback>\n"
        )
        evaluator_criteria = [{"role": "user", "content": evaluator_criteria}]
        messages = evaluator_task + evaluator_context + evaluator_criteria
        return messages
    
    def generate_refine_prompt(
        self,
        data: pd.Series,
        answer: Any,
        user_instruction: str,
        feedback: str,
        dtypes: list[str]
    ):
        initial_generate_prompt = self.generate_prompt(data, user_instruction, dtypes)
        refine_context =(
            "There is feedback from your previous output.\n"
            f"Output:\n{answer}\nFeedback:\n{feedback}\n"
            "Your task is to refine the output based on the feedback."
        )
        refine_prompt = [{"role": "user", "content": refine_context}]
        messages = initial_generate_prompt + refine_prompt
        return messages
