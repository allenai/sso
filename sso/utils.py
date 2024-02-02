from __future__ import annotations
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from sso.trajectory import State

from functools import lru_cache
import numpy as np

from sso.llm import get_embedding


@lru_cache(maxsize=100000)
def clean_feature(feature: str, remove_fill_words=False) -> str:
    words_to_skip = set(["to", "the", "for", "on", "in", "a", "an", ""]) if remove_fill_words else set([""])
    words = []
    for word in feature.lower().split():
        word = word.strip(".,!?\"')(}{][:; \t\n")
        if word not in words_to_skip:
            words.append(word)
    return " ".join(words)


@lru_cache(maxsize=100000)
def _get_emedding_similarity(text1: str, text2: str) -> float:
    if clean_feature(text1, remove_fill_words=True) == clean_feature(text2, remove_fill_words=True):
        return 1.0
    embedding1 = np.array(get_embedding(text1))
    embedding2 = np.array(get_embedding(text2))
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def get_similarity(text1: str, text2: str) -> float:
    return _get_emedding_similarity(text1, text2)


def get_feature_similarity(feature: str, features: List[str]) -> float:
    if isinstance(features, str):
        features = [features]
    return max(get_similarity(feature, feature2) for feature2 in features)


def get_similar_action(action: str, actions: List[str]) -> str:
    for x in actions:
        if clean_feature(action, remove_fill_words=True) == clean_feature(x, remove_fill_words=True):
            return x
    return None


def get_state_similarity(state1: State, state2: State, init_state: bool = False) -> float:
    if state1.last_action is None and state2.last_action is None or init_state:
        return _get_emedding_similarity(state1.state_description, state2.state_description)
    elif state1.last_action is not None and state2.last_action is not None:
        return _get_emedding_similarity(
            "You chose to {}.\n\n".format(state1.last_action) + state1.state_description,
            "You chose to {}.\n\n".format(state2.last_action) + state2.state_description
        )
    else:
        return 0
