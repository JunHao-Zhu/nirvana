from typing import Any, List, Dict, Union


class FilterPrompter:
    def __init__(self):
        self.system_instruction = (
            "You are a helpful assistant helping the user make sense of their data. "
            "You are performing a filter operation (one input: True or False) to "
            "determine whether the given data satisfies the given condition(s)."
        )
        self.output_format = (
            "The answer should strictly be either True or False. "
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
        sys_message = [
            {"role": "system", "content": self.system_instruction},
            {"role": "system", "content": self.output_format}
        ]

        # 2. Prepare data
        user_content = []
        if dtype == "str":
            user_content.append({"type": "input_text", "text": data})
        elif dtype == "image":
            user_content.append(
                {"type": "input_image", "image_url": data}
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
            data: Any,
            user_instruction: str,
            dtype: str,
            demos: List[Dict[str, Any]]
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
            if dtype == str:
                demo_content = [
                    {"type": "input_text", "text": demo["data"]},
                    {"type": "input_text", "text": user_instruction},
                    {"type": "input_text", "text": demo["answer"]}
                ]
            elif dtype == "image":
                demo_content = [
                    {"type": "input_image", "image_url": demo["data"]},
                    {"type": "input_text", "text": user_instruction},
                    {"type": "input_text", "text": demo["answer"]}
                ]
            else:
                raise ValueError(f"Data type {dtype} is not supported.")
            demos_message.append(
                {"role": "assistant", "content": demo_content}
            )

        # 3. Prepare user message
        if dtype == "str":
            user_content = [
                {"type": "input_text", "text": data},
                {"type": "input_text", "text": user_instruction}
            ]
        elif dtype == "image":
            user_content = [
                {"type": "input_image", "image_url": data},
                {"type": "input_text", "text": user_instruction}
            ]
        else:
            raise ValueError(f"Data type {dtype} is not supported.")
        user_message = [{"role": "user", "content": user_content}]
        
        messages = sys_message + demos_message + user_message
        return messages
    
    def generate_evaluate_prompt(
            self,
            data: Any,
            answer: Any,
            user_instruction: str,
            dtype: str = "str"
    ):
        evaluator_task = [{"role": "user", "content": "Analyze the filter operation tasked with evaluating the condition on the given data:"}]
        instruction = user_instruction.strip() + " " + self.output_format
        if dtype == "str":
            evaluator_context = [
                {"role": "user", "content": f"Input:\n{data}\nOutput:\n{answer}\nInstruction:\n{instruction}"}
            ]
        elif dtype == "image":
            _input = [{"type": "input_text", "text": "Input:"}, {"type": "input_image", "image_url": data}]
            _output = [{"type": "input_text", "text": f"Output:\n{answer}"}]
            _instruction = [{"type": "input_text", "text": f"Instruction:\n{instruction}"}]
            evaluator_context = [
                {"role": "user", "content": _input + _output + _instruction}
            ]
        else:
            raise ValueError(f"Data type {dtype} is not supported.")

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
            data: Any,
            answer: Any,
            user_instruction: str,
            feedback: str,
            dtype: str = "str"
    ):
        initial_generate_prompt = self.generate_prompt(data, user_instruction, dtype)
        refine_context =(
            "There is feedback from your previous output.\n"
            f"Output:\n{answer}\nFeedback:\n{feedback}\n"
            "Your task is to refine the output based on the feedback."
        )
        refine_prompt = [{"role": "user", "content": refine_context}]
        messages = initial_generate_prompt + refine_prompt
        return messages
