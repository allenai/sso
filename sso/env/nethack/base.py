from typing import Tuple, Dict, Any, List
import gym
import minihack
from nle.nethack.actions import *
from nle_language_wrapper import NLELanguageWrapper
from sso.env.nethack.utils import ACTION_TEMPLATES

import sso.env.nethack.maps
from sso.env.nethack.utils import get_message, get_lang_obs
from sso.env import Env
from sso.trajectory import State


class NetHackTask(Env):

    def __init__(
            self,
            task: str = None,
            max_steps: int = 100,
            **kwargs
        ):
        assert task is not None, "You must specify a task to load"
        self.task = task
        self.env = gym.make(
                self.task,
                observation_keys=("glyphs", "blstats", "tty_chars", "inv_strs", "inv_letters", "tty_cursor")
            )
        self.lang_to_action = NLELanguageWrapper(self.env).pre_step

        self._max_steps = max_steps
        self.t = 0
        self.last_nle_obs = None

    @property
    def action_templates(self) -> List[str]:
        return ACTION_TEMPLATES

    def parse_action(self, action: str) -> Tuple[str, List[str]]:
        env_action = action.split("Next Action:")[-1].split("Next action:")[-1].split("next action:")[-1]
        env_action = env_action.strip(" \n\"'([{")
        env_action = env_action[0] if len(env_action) > 0 else env_action
        return env_action, [env_action]
    
    def get_action_prompt(self) -> str:
        res = "Below is a mapping of keys to common actions. Your generated action should always be a single character.\n\t"
        res += "\n\t".join(self.action_templates)
        return res

    def reset(self, **kwargs) -> Tuple[State, Dict[str, Any]]:
        obs = self.env.reset()
        self.t = 0
        self.last_nle_obs = obs
        info = dict(
            task_id=self.task,
            task_description="Task Description: " + self.env.TASK_DESCRIPTION,
            success=False
        )
        return self.get_state(obs, "", "", 0, False), info

    def step(self, action: str) -> Tuple[State, float, bool, Dict[str, Any]]:
        parsed_action, env_actions = self.parse_action(action)

        reward, done, info = 0, False, dict()
        try:
            for env_action in env_actions:
                self.last_nle_obs, r, done, info = self.env.step(self.lang_to_action(env_action))
                reward += r
                if done:
                    break
            state = self.get_state(self.last_nle_obs, action, parsed_action, reward, done)
        except ValueError:
            state = State(
                observation="Invalid action.",
                state_features=["Invalid action."],
                state_description="Invalid action.",
                last_generation=action,
                last_action=parsed_action,
                reward=reward,
                done=done,
                task_description="Task Description: " + self.env.TASK_DESCRIPTION,
                action_prompt=self.get_action_prompt(),
                all_templates=self.action_templates,
            )

        self.t += 1
        if self.t >= self.max_steps:
            done = True
        info.update(dict(
            task_id=self.task,
            task_description="Task Description: " + self.env.TASK_DESCRIPTION,
            success=info["end_status"] == 2 if "end_status" in info else False,
        ))

        return state, reward, done, info

    def get_state(self, obs, generation: str, action: str, reward: float, done: bool) -> State:
        return State(
            observation=get_message(obs),
            state_features=get_lang_obs(obs, as_list=True),
            state_description=get_lang_obs(obs, as_list=False),
            last_generation=generation,
            last_action=action,
            reward=reward,
            done=done,
            task_description="Task Description: " + self.env.TASK_DESCRIPTION,
            action_prompt=self.get_action_prompt(),
            all_templates=self.action_templates,
        )

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def num_train(self) -> int:
        return 1

    @property
    def num_test(self) -> int:
        return 1

    @property
    def train_ids(self) -> Tuple[str]:
        return (self.task,)

    @property
    def test_ids(self) -> Tuple[str]:
        return (self.task,)
