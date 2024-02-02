from __future__ import annotations
from typing import List, Union, Dict, Any
from copy import deepcopy
import numpy as np

from sso.utils import get_feature_similarity


class State:
    def __init__(
            self,
            observation: str = None,
            state_description: str = None,
            state_features: List[str] = None,
            last_generation: str = None,
            last_action: str = None,
            reward: float = 0,
            done: bool = False,
            action_prompt: str = None,
            all_templates: List[str] = None,
            task_description: str = None,
            skill_target: str = None
        ):
        self.observation = observation
        self.state_description = state_description
        self.state_features = state_features
        self.last_generation = last_generation
        self.last_action = last_action
        self.reward = reward
        self.done = done
        self.action_prompt = action_prompt
        self.all_templates = all_templates
        self.task_description = task_description
        self.skill_target = skill_target

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and \
            self.state_features == other.state_features and \
            self.last_action == other.last_action and \
            self.task_description == other.task_description

    def __hash__(self) -> int:
        return hash((tuple(self.state_features), self.last_action, self.task_description))

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            observation=self.observation,
            state_description=self.state_description,
            state_features=self.state_features,
            last_generation=self.last_generation,
            last_action=self.last_action,
            reward=self.reward,
            done=self.done,
            action_prompt=self.action_prompt,
            all_templates=self.all_templates,
            task_description=self.task_description,
            skill_target=self.skill_target
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> State:
        return State(
            observation=data["observation"],
            state_description=data["state_description"],
            state_features=data["state_features"],
            last_generation=data["last_generation"],
            last_action=data["last_action"],
            reward=data["reward"],
            done=data["done"],
            action_prompt=data["action_prompt"],
            all_templates=data["all_templates"],
            task_description=data["task_description"],
            skill_target=data["skill_target"]
        )

    def state_similarity(self, state: State) -> float:
        return get_feature_similarity(self.state_description, state.state_description)

    def action_similarity(self, state: State) -> float:
        if self.last_action is not None and state.last_action is not None:
            return get_feature_similarity(self.last_action, state.last_action)
        elif self.last_generation is None and state.last_generation is None:
            return 1
        else:
            return 0


class Trajectory:
    def __init__(self, steps: List[State] = None, task_description: str = None):
        self.steps = steps if steps is not None else []
        self.task_description = task_description

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> State:
        return self.steps[index]
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Trajectory) and self.steps == other.steps and \
            self.task_description == other.task_description
    
    def __hash__(self):
        return hash(tuple(self.steps + [self.task_description]))

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            task_description=self.task_description,
            steps=[x.to_dict() for x in self.steps]
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Trajectory:
        return Trajectory(
            steps=[State.from_dict(x) for x in data["steps"]],
            task_description=data["task_description"]
        )

    def slice(self, start: int, end: int) -> Trajectory:
        sliced = Trajectory(
            deepcopy(self.steps[start:end]),
            task_description=self.task_description
        )
        return sliced

    def insert(self, state: State):
        self.steps.append(state)

    def build_string(
            self,
            use_task: bool = True,
            use_features: bool = True,
            use_actions: bool = True,
            start_idx: int = 0,
            end_idx: int = None,
            trim_state: bool = True,
            as_list: bool = False
        ) -> Union[str, List[str]]:

        return Trajectory.build_string_from_states(
            self.steps,
            self.task_description if use_task else None,
            use_features,
            use_actions,
            start_idx,
            end_idx,
            trim_state,
            as_list
        )

    @staticmethod
    def build_string_from_states(
            states: List[State],
            task_description: str = None,
            use_features: bool = True,
            use_actions: bool = True,
            start_idx: int = 0,
            end_idx: int = None,
            trim_state: bool = True,
            as_list: bool = False
        ) -> Union[str, List[str]]:

        res = []
        last_feats = []
        next_step = ""
        if task_description:
            next_step += task_description + "\n\n"

        for i, step in enumerate(states[start_idx:end_idx]):
            next_step += "Step #{}:\n".format(i + 1)

            next_step += step.observation
            if trim_state:
                if use_features and "you see" not in step.observation.lower():
                    new_feats = []
                    for feat in sorted(step.state_features, key=lambda x: len(x), reverse=True):
                        if feat not in last_feats and feat != step.observation \
                                and not any(feat.replace(" in your inventory", "") in x for x in new_feats):
                            new_feats.append(feat)
                    if len(new_feats) > 0:
                        next_step += "\nYou also observe:\n\t"
                        next_step += "\n\t".join(new_feats)
                last_feats = step.state_features
            elif use_features and "you see" not in step.observation.lower():
                next_step += "\n" + step.state_description

            if use_actions and i + 1 < len(states):
                next_step += "\nYou choose to: {}".format(states[i + 1].last_action)

            res.append(next_step)
            next_step = ""

        return res if as_list else "\n\n".join(res)

    def similarity(self, trajectories: Union[Trajectory, List[Trajectory]], length_diff_penalty: float = 0.4) -> float:
        if isinstance(trajectories, Trajectory):
            trajectories = [trajectories]

        state_similarity = 1
        action_similarity = 1
        for trajectory in trajectories:
            assert len(self) == len(trajectory)

            state_similarity = min(
                state_similarity,
                np.mean([self[i].state_similarity(trajectory[i]) for i in range(len(self))])
            )
            action_similarity = min(
                action_similarity,
                np.mean([self[i].action_similarity(trajectory[i]) for i in range(1, len(self))])
            )

        return state_similarity, action_similarity
