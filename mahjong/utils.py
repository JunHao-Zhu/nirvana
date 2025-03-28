from copy import deepcopy
from pathlib import Path
from typing import Union

from mahjong.models.llm_backbone import LLMClient
from mahjong.ops.base import BaseOperation

    
def configure_llm_backbone(
        model_name: str = None, api_key: Union[str, Path] = None, base_url=None
):
    """
    Configures the LLM (Large Language Model) backbone by initializing and setting it 
    for use in the application.
    Args:
        model_name (str, optional): The name of the LLM model to be used. Defaults to None.
        api_key (Union[str, Path], optional): The API key or path to the API key required 
            to authenticate with the LLM service. Defaults to None.
        base_url (str, optional): The base URL of the LLM service endpoint. Defaults to None.
    Returns:
        None

    Example:
        ```python
        >>> import mahjong as mjg
        >>> mjg.configure_llm_backbone(model_name="gpt-3.5-turbo", api_key="your-api-key")
        ```
    """
    llm_client = LLMClient.configure(model_name, api_key, base_url)
    BaseOperation.set_llm(deepcopy(llm_client))
