from typing import Any, Union, Iterable


class SummarizePrompter:
    def __init__(self):
        self.instruction =  (
            "A table has the following columns:\n/*\n{columns}\n*/\n"
            "Describe briefly what the {column} column represents. If not possible, simply state 'No description'."
        )

    def generate_prompt(
            self,
            column: str,
            columns: Union[str, Iterable[str]],
    ):
        # 1. Prepare message
        prompt = self.instruction.format(
            columns=" | ".join(columns) if isinstance(columns, Iterable) else columns,
            column=column
        )
        message = [{"role": "user", "content": prompt}]
        return message


class RerankPrompter:
    def __init__(self):
        self.content_instruction = (
            "Given a table with the following columns:\n*/\n{columns}\n*/\n"
            "and this question:\n*/{query}\n*/\n"
            "Is the table relevant to answer the question? The answer should strictly be either True or False.\n"
            "Output the answer concisely in the following format.\n"
            "<output> True or False </output>\n"
        )
        self.context_instruction = (
            "Given this context describing a table:\n*/\n{context}\n*/\n"
            "and this question:\n*/{query}\n*/\n"
            "Is the table relevant to answer the question? The answer should strictly be either True or False.\n"
            "Output the answer concisely in the following format.\n"
            "<output> True or False </output>\n"
        )
    
    def generate_prompt(
            self,
            query: str,
            columns_desc: str = None,
            context_desc: str = None,
    ):
        # 1. Prepare message
        if columns_desc:
            prompt = self.content_instruction.format(columns=columns_desc, query=query)
        elif context_desc:
            prompt = self.context_instruction.format(context=context_desc, query=query)
        else:
            raise ValueError("Either columns_desc or context_desc should be provided.")
        message = [{"role": "user", "content": prompt}]
        return message
