from typing import List, Tuple, Union

import os
import json
from tqdm import tqdm

from sso.trajectory import Trajectory
from sso.skill import Skill


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
    
    def get_memories(self, trajectory: Trajectory = None, n: int = None) -> List[Union[Trajectory, Skill]]:
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
