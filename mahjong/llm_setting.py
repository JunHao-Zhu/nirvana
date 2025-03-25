import re
from pathlib import Path
from typing import Union, Optional
from openai import OpenAI

    
def _get_api_key_from_file(self, file):
    with open(file, 'r') as api_file:
        api_key = api_file.readline()
    return api_key


def _create_client(self, api_key, **kwargs):
    assert api_key != "", "API key is required."
    client = OpenAI(api_key=api_key, base_url=kwargs["base_url"])
    return client


class LLMClient:
    model: Optional[str] = None
    client = None

    @classmethod
    def configure(cls, model_name: str = None, api_key: Union[str, Path] = None, base_url=None):
        cls.model = model_name
        api_key = api_key if isinstance(api_key, str) else _get_api_key_from_file(api_key)
        cls.client = _create_client(cls.api_key, base_url=base_url)
    
    def __call__(
            self,
            prompt: str,
            system_prompt: str = "",
            max_tokens: int = 256,
            temperature: float = 0.1,
            **kwargs,
    ):
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content

    def _extract_xml(self, text: str, tag: str):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ''
    

client = LLMClient()
