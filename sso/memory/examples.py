from typing import List

from sso.trajectory import Trajectory
from sso.memory import Memory


class ExamplesMemory(Memory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_reward = float("-inf")

    def insert(self, trajectory: Trajectory) -> None:
        reward = sum(x.reward for x in trajectory if x.reward is not None)
        if reward >= self.best_reward:
            self.trajectories.append(trajectory)
            self.best_reward = max(reward, self.best_reward)
            best_trajectories = []
            for traj in self.trajectories:
                if sum(x.reward for x in traj if x.reward is not None) == self.best_reward:
                    best_trajectories.append(traj)
            self.trajectories = best_trajectories
    
    def get_memories(self, trajectory: Trajectory = None, n: int = None) -> List[Trajectory]:
        if n is None:
            n = len(self.trajectories)
        return self.trajectories[-n:]
