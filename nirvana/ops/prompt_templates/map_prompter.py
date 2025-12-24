import pandas as pd
from typing import Any, List, Dict


class MapPrompter:
    def __init__(self):
        self.system_instruction = (
            "You are a helpful assistant helping the user make sense of their data. "
            "You are performing a map operation to project the given data into correct values for a set of output fields based on the user's instruction."
        )
        self.output_format: str = None

    def prepara_output_format(self, output_columns: list[str]):
        output_format = "Output the result of the map operation (each field) concisely in the following format.\n"
        for column_name in output_columns:
            output_format += f"<{column_name}> fill in correct value </{column_name}>\n"
        return output_format

    def generate_prompt(
        self, 
        data: pd.Series,
        user_instruction: str,
        output_columns: list[str],
        dtypes: list[str],
    ):
        """
        Generates a prompt message for LLMs based on user instructions 
        and provided data.

        Args:
            user_instruction (str): The instruction or query provided by the user.
            data (Any): The data to be included in the prompt. It can be a string or an 
                instance of `mjg.ImageDtype`. If the data type is unsupported, a ValueError 
                is raised.

        Returns:
            list: A list of dictionaries representing the structured prompt messages. 
            The messages include a system message and a user message, where the user 
            message contains the provided data and user instruction.

        Raises:
            ValueError: If the type of `data` is not supported.
        """
        # 1. Prepare system message
        if self.output_format is None:
            self.output_format = self.prepara_output_format(output_columns=output_columns)
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
        
        # 3. Prepare the given transformation instruction
        user_content.append({"type": "input_text", "text": user_instruction})
        user_message = [{"role": "user", "content": user_content}]

        messages = sys_message + user_message
        return messages
    
    def generate_fewshot_prompt(
        self,
        data: pd.Series,
        user_instruction: str,
        output_columns: list[str],
        dtypes: list[str],
        demos: list[dict[str, Any]]
    ):
        """
        Generates a prompt with demonstration examples for an LLM.
        Args:
            user_instruction (str): The instruction provided by the user to guide the AI's response.
            data (Any): The input data provided by the user. Must be a string; otherwise, a ValueError is raised.
            demos (List[Dict[str, Any]]): A list of demonstration examples. Each example should be a dictionary 
                containing the keys "data" (input data as a string) and "answer" (expected output as a string). 
                If a demo is not a string, a ValueError is raised.
        Returns:
            List[Dict[str, Any]]: A list of messages formatted for the conversational AI system. The messages 
            include:
                - A system message containing the system's instruction.
                - Demonstration messages based on the provided examples.
                - A user message containing the input data and user instruction.
        Raises:
            ValueError: If the type of `data` or `demo["data"]` is not supported.
        """
        # 1. Prepare system message
        if self.output_format is None:
            self.output_format = self.prepara_output_format(output_columns=output_columns)
        sys_message = [
            {"role": "system", "content": self.system_instruction},
            {"role": "system", "content": self.output_format}
        ]

        # 2. Prepare demonstration message
        demos_message = [{"type": "input_text", "text": f"Here are some examples:"}]
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
        
        # 4. Prepare the given condition
        user_content.append({"type": "input_text", "text": user_instruction})
        user_message = [{"role": "user", "content": user_content}]
        
        messages = sys_message + demos_message + user_message
        return messages
    
    def generate_evaluate_prompt(
        self,
        data: Any,
        answer: Any,
        user_instruction: str,
        dtypes: list[str],
    ):
        evaluator_task = [{"role": "user", "content": "Analyze the map operation tasked with transforming data based on a given instruction:"}]
        
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
            "Evaluate the map operation based on the following criteria:\n"
            "1. Does the output strictly adhere to the required output format?\n"
            "2. Does the output satisfy the instruction?\n"
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
        data: Any,
        answer: Any,
        user_instruction: str,
        output_columns: list[str],
        feedback: str,
        dtypes: list[str],
    ):
        initial_generate_prompt = self.generate_prompt(data, user_instruction, output_columns, dtypes)
        refine_context =(
            "There is feedback from your previous output.\n"
            f"Output:\n{answer}\nFeedback:\n{feedback}\n"
            "Your task is to refine the output based on the feedback."
        )
        refine_prompt = [{"role": "user", "content": refine_context}]
        messages = initial_generate_prompt + refine_prompt
        return messages
