from typing import Any, List, Dict


class MapPrompter:
    def __init__(self):
        self.system_instruction = (
            "You are a helpful assistant helping the user make sense of their data. "
            "You are performing a map operation (one input: one output) to "
            "project the given data based on the user's instruction.\n"
            "Output the result of the map operation concisely in the following format.\n"
            "<output> LLM output </output>\n"
        )

    def generate_prompt(
            self, 
            data: Any,
            user_instruction: str,
            dtype: str = "str"
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
        sys_message = [{"role": "system", "content": self.system_instruction}]

        # 2. Prepare user message
        if dtype == "str":
            user_content = [{"type": "text", "text": data}]
        elif dtype == "image":
            user_content = [
                {"type": "image", "image_url": {"url": data}}
            ]
        else:
            raise ValueError(f"Data type {type(data)} is not supported.")
        user_content.append({"type": "text", "text": user_instruction})
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
        """
        Generates a Chain-of-Thought (CoT) prompt for an LLM.
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
        sys_message = [{"role": "system", "content": self.system_instruction}]

        # 2. Prepare demonstration message
        demos_message = []
        for demo in demos:
            demo_content = []
            if dtype == "str":
                demo_content = [
                    {"type": "text", "text": demo["data"]},
                    {"type": "text", "text": user_instruction},
                    {"type": "text", "text": demo["answer"]}
                ]
            elif dtype == "image":
                demo_content = [
                    {"type": "image", "image_url": {"url": demo["data"]}},
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
                {"type": "image", "image_url": {"url": data}},
                {"type": "text", "text": user_instruction}
            ]
        else:
            raise ValueError(f"Data type {dtype} is not supported.")
        user_message = [{"role": "user", "content": user_content}]
        
        messages = sys_message + demos_message + user_message
        return messages
