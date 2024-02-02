from typing import List, Dict
import os
import time
from functools import lru_cache
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_response(model: str, messages: List[Dict[str, str]], max_tries=50, temperature=1, **kwargs) -> str:
    completion = None
    num_tries = 0
    while not completion and num_tries < max_tries:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                **kwargs
            ).choices[0].message.content
            break
        except Exception as e:
            num_tries += 1
            print("try {}: {}".format(num_tries, e))
            if "maximum context length" in str(e):
                if len(messages) > 3:
                    if messages[0]["role"] == "system":
                        messages = [messages[0]] + messages[3:]
                    else:
                        messages = messages[2:]
                else:
                    raise RuntimeError("messages too long")
            time.sleep(2)
    if not completion:
        raise RuntimeError("Failed to get response from API")
    return completion


@lru_cache(maxsize=1000)
def get_embedding(model: str, content: str, max_tries=50) -> List[float]:
    embedding = None
    num_tries = 0
    while not embedding and num_tries < max_tries:
        try:
            embedding = openai.Embedding.create(model=model, input=content).data[0].embedding
            break
        except Exception as e:
            num_tries += 1
            print("try {}: {}".format(num_tries, e))
            time.sleep(2)
    if not embedding:
        raise RuntimeError("Failed to get embedding response from API")
    return embedding
