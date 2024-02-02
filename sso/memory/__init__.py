from typing import List, Tuple, Union

import os
import json
import numpy as np
from tqdm import tqdm
from functools import lru_cache

from sso.trajectory import Trajectory
from sso.skill import Skill
from sso.llm import query_llm
from sso.utils import get_state_similarity


class Memory:

    def __init__(
            self,
            max_trajectories: int = 10,
            skill_memory_scale: float = 2.0,
            max_retrieval: int = 3,
            discount: float = 0.9,
        ):
        self.max_trajectories = max_trajectories
        self.skill_memory_scale = skill_memory_scale
        self.max_retrieval = max_retrieval
        self.discount = discount

        self.trajectories: List[Trajectory] = []
        self.memory: List[Skill] = []
        self.sampled: List[Skill] = []
        self.skill_feedback: List[List[Tuple[Skill, float, bool]]] = [[]]

    def insert(self, trajectory: Trajectory) -> Trajectory:
        raise NotImplementedError

    def build(self, trajectories: Union[Trajectory, List[Trajectory]] = [], **kwargs) -> None:
        if isinstance(trajectories, Trajectory):
            trajectories = [trajectories]
        print("Inserting trajectories...")
        for trajectory in tqdm(trajectories):
            self.insert(trajectory)

    def clear(self) -> None:
        self.trajectories: List[Trajectory] = []
        self.memory: List[Skill] = []
        self.sampled: List[Skill] = []
        self.skill_feedback: List[List[Tuple[Skill, float, bool]]] = [[]]

    def log_sampled(self, step: int, skill: Skill) -> None:
        if not any(skill == x for _, x in self.sampled):
            self.sampled.append((step, skill))

    @staticmethod
    @lru_cache(maxsize=1000)
    def _check_similar_skills(memory: Tuple[Skill], skill: Skill) -> bool:
        if len(memory) == 0:
            return False
        system_message = "You are an expert planning system capable of completing various tasks in this environment. The environment provides observations in response to your actions. You have a library of skills that you reference to execute actions. A skill is composed of a list of instructions, a list of prerequisites, and a target state."

        prompt = "Given the below list of existing skills and the new skill, determine which existing skills are semantically equivalent to the new skill. Output a comma delimited list of numbers that correspond to the existing skills. If no skills are equivalent output 'None'. Use the following format for your response:"
        prompt += "\nThe existing skills permit the agent to... The new skill will permit the agent to...\nEquivalent skills: 1, 2, 3"
        prompt += "\n\nExisting Skills:"
        for i, existing_skill in enumerate(memory):
            prompt += "\nSkill {} prerequisites: {}".format(i + 1, ", ".join(existing_skill.prereqs))
            prompt += "\nSkill {} target: {}".format(i + 1, existing_skill.target)
            prompt += "\nSkill {} instructions: {}".format(i + 1, ", ".join(existing_skill.instructions))
        prompt += "\n\nNew skill prerequisites: {}".format(", ".join(skill.prereqs))
        prompt += "\nNew skill target: {}".format(skill.target)
        prompt += "\nNew skill instructions: {}".format(", ".join(skill.instructions))

        messages = [dict(role="system", content=system_message), dict(role="user", content=prompt)]
        response = query_llm(messages, temperature=0).lower()
        return "none" not in response.split("skills:")[-1].split("skill:")[-1]

    def check_similar_skills(self, skill: Skill) -> bool:
        return Memory._check_similar_skills(tuple(self.memory), skill)

    def get_similar_skills(self, trajectory: Trajectory) -> List[Skill]:
        retrieved_skills = []

        unused = self.memory.copy()
        for state in trajectory:
            for skill in unused:
                if state.skill_target is not None and state.skill_target == skill.target:
                    retrieved_skills.append(skill)
                    unused.remove(skill)
                    break

        unused = sorted(
            unused,
            key=lambda x: np.mean([
                get_state_similarity(subtraj[0], trajectory[-1], init_state=True)
                for subtraj in x.trajectories
            ])
        )

        retrieved_skills += unused[-self.max_retrieval:]

        return list(set(retrieved_skills))

    def save(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "skill_info.json"), "w") as f:
            json.dump([
                s.info() for s in self.memory
            ], f, indent=4)
        with open(os.path.join(save_path, "memory.json"), "w") as f:
            json.dump(dict(
                trajectories=[s.to_dict() for s in self.trajectories],
                skills=[s.to_dict() for s in self.memory],
                skill_feedback=[[[r, x, s.to_dict()] for s, r, x in episode] for episode in self.skill_feedback],
            ), f, indent=4)

    def load(self, load_path: str, rebuild: bool = False) -> None:
        with open(os.path.join(load_path, "memory.json"), "r") as f:
            data = json.load(f)
        self.trajectories = [Trajectory.from_dict(x) for x in data["trajectories"]]
        if rebuild:
            self.build()
        else:
            self.memory = []
            for skill_data in data["skills"]:
                skill = Skill.from_dict(skill_data)
                skill.build()
                self.memory.append(skill)
            self.skill_feedback = [[]]
            for episode in data["skill_feedback"]:
                self.skill_feedback.append([])
                for r, x, skill_data in episode:
                    skill = Skill.from_dict(skill_data)
                    skill.build()
                    self.skill_feedback[-1].append((skill, r, x))
