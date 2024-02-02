from typing import List, Dict

from sso.llm.gpt import get_response as get_response_gpt, get_embedding as get_embedding_gpt


global DEFAULT_MODEL
DEFAULT_MODEL = None

global DEFAULT_TEMP
DEFAULT_TEMP = None

global DEFAULT_EMBEDDING
DEFAULT_EMBEDDING = None


def set_default_model(model: str = None, temp: float = None, embedding: str = None) -> None:
    if model is not None:
        global DEFAULT_MODEL
        DEFAULT_MODEL = model

    if temp is not None:
        global DEFAULT_TEMP
        DEFAULT_TEMP = temp

    if embedding is not None:
        global DEFAULT_EMBEDDING
        DEFAULT_EMBEDDING = embedding


def query_llm(messages: List[Dict[str, str]], model: str = None, temperature: float = 1, **generation_kwargs) -> str:
    if model is None:
        global DEFAULT_MODEL
        model = DEFAULT_MODEL
    if temperature is None:
        global DEFAULT_TEMP
        temperature = DEFAULT_TEMP
    if model.startswith("gpt"):
        return get_response_gpt(model, messages, temperature=temperature, **generation_kwargs)


def get_embedding(content: str, model: str = None) -> List[float]:
    if model is None:
        global DEFAULT_EMBEDDING
        model = DEFAULT_EMBEDDING

    if model.startswith("text"):
        return get_embedding_gpt(model, content)
