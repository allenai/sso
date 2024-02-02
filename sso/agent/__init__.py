from typing import Any, List, Dict

from sso.trajectory import Trajectory
from sso.memory import Memory


class Agent:

    def __init__(self, memory: Memory = None, max_history: int = 10, trim_states: bool = True):
        assert memory is not None
        self.memory = memory
        self._log = []
        self.freeze_memory = False
        self.max_history = max_history
        self.trim_states = trim_states

    def train(self, train: bool = True) -> None:
        self.freeze_memory = not train

    def clear(self) -> None:
        self.memory.clear()

    def act(self, trajectory: Trajectory) -> str:
        self.step_log()
        self.log("reward", trajectory[-1].reward)
        self.log("last_action", trajectory[-1].last_action)
        self.log("state_description", trajectory[-1].state_description)

        # Get next action
        return None

    def record_done(self, trajectory: Trajectory) -> None:
        self.step_log()
        self.log("reward", trajectory[-1].reward)
        self.log("last_action", trajectory[-1].last_action)
        self.log("state_description", trajectory[-1].state_description)

        if not self.freeze_memory:
            self.memory.build(trajectory)

    def save(self, save_dir: str) -> None:
        self.memory.save(save_dir)

    def load(self, load_dir: str, rebuild: bool = False) -> None:
        self.memory.load(load_dir, rebuild=rebuild)

    def log(self, key: str, value: Any) -> None:
        if len(self._log) == 0:
            self.step_log()
        if key in self._log[-1]:
            if isinstance(self._log[-1][key], list):
                self._log[-1][key].append(value)
            else:
                self._log[-1][key] = [self._log[-1][key], value]
        else:
            self._log[-1][key] = value

    def step_log(self) -> None:
        self._log.append(dict())

    def reset_log(self) -> None:
        self._log = []

    def get_log(self) -> List[Dict[str, Any]]:
        return self._log
