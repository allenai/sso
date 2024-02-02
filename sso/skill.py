from __future__ import annotations
from typing import List, Dict, Any

import numpy as np
from copy import deepcopy

from sso.llm import query_llm
from sso.trajectory import Trajectory


class Skill:

    def __init__(self, instructions: List[str] = None, target: str = None, max_trajectories: int = 5):
        self.instructions = instructions
        self.target = target
        self.max_trajectories = max_trajectories
        self.trajectories: List[Trajectory] = []
        self.traj_indices: List[int] = []
        self.start_indices: List[int] = []
        self.end_indices: List[int] = []
        self.prereqs = None
        self.state_similarity = 1
        self.action_similarity = 1

    def __str__(self):
        return "Trajectory Count: {}\nTarget: {}\nInstructions:\n\t{}".format(
            len(self.trajectories),
            self.target,
            "\n\t".join(self.instructions)
        )

    def __eq__(self, other: object):
        return isinstance(other, Skill) and \
            set(self.trajectories[-self.max_trajectories:]) == set(other.trajectories[-self.max_trajectories:])

    def __hash__(self):
        return hash(tuple(set(self.trajectories[-self.max_trajectories:])))

    def traj_len(self):
        return min(len(x) for x in self.trajectories)

    def traj_count(self):
        return len(self.trajectories)

    def step_count(self):
        return sum(len(x) - 1 for x in self.trajectories)

    def reward(self) -> float:
        return np.mean([np.sum([x.reward for x in traj]) for traj in self.trajectories])

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        skill = Skill()
        skill.trajectories = [Trajectory.from_dict(x) for x in data["trajectories"]]
        skill.instructions = data["instructions"]
        skill.target = data["target"]
        skill.prereqs = data["prereqs"]
        skill.state_similarity = data["state_similarity"]
        skill.action_similarity = data["action_similarity"]
        skill.traj_indices = data["traj_indices"]
        skill.start_indices = data["start_indices"]
        skill.end_indices = data["end_indices"]
        return skill

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            instructions=self.instructions,
            target=self.target,
            prereqs=self.prereqs,
            state_similarity=self.state_similarity,
            action_similarity=self.action_similarity,
            traj_indices=self.traj_indices,
            start_indices=self.start_indices,
            end_indices=self.end_indices,
            trajectories=[x.to_dict() for x in self.trajectories]
        )

    def info(self) -> Dict[str, Any]:
        return dict(
            instructions=self.instructions,
            target=self.target,
            prereqs=self.prereqs,
            state_similarity=self.state_similarity,
            action_similarity=self.action_similarity,
            rewards=[float(np.sum([x.reward for x in traj])) for traj in self.trajectories],
            actions=[[step.last_action for step in traj[1:]] for traj in self.trajectories],
            traj_indices=self.traj_indices,
            start_indices=self.start_indices,
            end_indices=self.end_indices,
        )

    def try_add(self, trajectory: Trajectory, traj_idx: int, start_idx: int, end_idx: int) -> bool:
        if self.traj_count() <= 1:
            self.add(trajectory, traj_idx, start_idx, end_idx)
            return True
        if trajectory not in self.trajectories:
            state_sim, action_sim = trajectory.similarity(self.trajectories)
            if state_sim >= self.state_similarity and action_sim >= self.action_similarity:
                self.add(trajectory, traj_idx, start_idx, end_idx)
                return True
        return False

    def add(self, trajectory: Trajectory, traj_idx: int, start_idx: int, end_idx: int) -> Skill:
        if trajectory not in self.trajectories:
            if len(self.trajectories) > 0:
                state_sim, action_sim = trajectory.similarity(self.trajectories)
                self.state_similarity = min(self.state_similarity, state_sim)
                self.action_similarity = min(self.action_similarity, action_sim)
            self.trajectories.append(trajectory)
            self.traj_indices.append(traj_idx)
            self.start_indices.append(start_idx)
            self.end_indices.append(end_idx)
            self.instructions = None
            self.target = None
            self.prereqs = None
        return self

    def is_compatible(self, other: Skill) -> bool:
        shared = [
            (self.traj_indices.index(x), other.traj_indices.index(x))
            for x in self.traj_indices
            if x in other.traj_indices
        ]
        for (this_idx, other_idx) in shared:
            if range(
                max(self.start_indices[this_idx], other.start_indices[other_idx]),
                min(self.end_indices[this_idx], other.end_indices[other_idx]) + 1
            ):
                return False
        return True

    def _generate(self) -> str:
        system_message = "You are an expert planning system. You are creating reusable skills to execute when completing various tasks. You create skills by looking at successful examples of task completions. A skill is composed of a list of instructions and a target state. After creating a skill, it will be used to execute actions in an environment. The environment will return a set of observations that summarize the new environment state. These observations will be used in conjunction with the skill's target state to determine whether the last skill was successful."

        summary_prompt = "Consider the example trajectories of states and actions below. You'll be asked to analyze the similarities between each. Pay attention to the wording of the state observations and actions. Then you'll be asked to generate the common instructions, and target state for them."
        for t, traj in enumerate(self.trajectories[-self.max_trajectories:]):
            summary_prompt += "\n\nExample {}:".format(t + 1)
            summary_prompt += "\n\nInitial State: "
            summary_prompt += traj[0].state_description
            summary_prompt += "\n\nTrajectory:"
            for s, step in enumerate(traj[1:]):
                summary_prompt += "\nAction {}: {}".format(s, step.last_action)
                summary_prompt += "\nObservation {}: {}".format(s, step.observation)
            summary_prompt += "\n\nFinal State: "
            summary_prompt += traj[-1].state_description
        summary_prompt += "\n\nGenerate a summary of what is happening in the examples above and the similarities between them. Provide a name for the skill that is being executed in the examples above. Do not generate skill instructions or target yet."
        messages = [dict(role="system", content=system_message), dict(role="user", content=summary_prompt)]
        summary_response = query_llm(messages, temperature=0.7)

        instruction_prompt = "Generate a numbered list of instructions for completing the skill. The instructions should be similar to the actions in the examples. Instructions should use the action templates provided below. Create generic instructions that would be valid for every example but specific enough to be useful in the examples. Do not mention the examples in the instructions. Use the output format:\nSkill [skill name] instructions:\n1. instruction 1\n2. instruction 2\n..."
        instruction_prompt += "\n\nAction templates: {}".format(", ".join(self.trajectories[0][0].all_templates))
        messages += [dict(role="assistant", content=summary_response), dict(role="user", content=instruction_prompt)]
        instruction_response = query_llm(messages, temperature=0.7)

        target_prompt = "Generate a single target observation that would indicate the success of the skill. The target should be similar to one of the observations in the final states. Create a generic target that would be valid for every example. Do not mention the examples in the target. Use the output format:\nSkill [skill name] target: [target observation]"
        messages += [dict(role="user", content=target_prompt)]
        target_response = query_llm(messages, temperature=0.7)

        instructions = []
        for x in instruction_response.lower().split("instructions:")[-1].split("\n"):
            if len(x.strip()) > 0:
                if x.strip()[0].isnumeric():
                    instructions.append(x.lstrip(" \t)-.1234567890"))
                elif len(instructions) > 0:
                    instructions[-1] += " " + x.lstrip(" \t)-.1234567890")
        target = target_response.lower().replace("target observation", "").split("target:")[-1].strip(".-:[]\n\t ")
        return [], instructions, target

    def build(self, force: bool = False) -> bool:
        self.trajectories = [deepcopy(traj) for traj in self.trajectories]
        self.trajectories = sorted(self.trajectories, key=lambda x: len(x))

        if force or self.instructions is None or self.target is None or self.prereqs is None:
            self.prereqs, self.instructions, self.target = self._generate()
