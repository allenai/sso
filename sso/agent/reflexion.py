from typing import List
import os
import json

from sso.trajectory import Trajectory
from sso.llm import query_llm
from sso.agent import Agent
from sso.memory import Memory


class ReflexionAgent(Agent):

    def __init__(self, max_reflections: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.memory = Memory()  # Dummy memory
        self.max_reflections = max_reflections
        self.reflections = []

    def act(self, trajectory: Trajectory) -> str:
        super().act(trajectory)
        return self._act(trajectory)

    def save(self, save_dir: str, **kwargs) -> None:
        with open(os.path.join(save_dir, "reflections.json"), "w") as f:
            json.dump(self.reflections, f, indent=4)

    def load(self, load_dir: str, **kwargs) -> None:
        with open(os.path.join(load_dir, "reflections.json"), "r") as f:
            self.reflections = json.load(f)

    def record_done(self, trajectory: Trajectory) -> None:
        self.step_log()
        self.log("reward", trajectory[-1].reward)
        self.log("last_action", trajectory[-1].last_action)
        self.log("state_description", trajectory[-1].state_description)

        if not self.freeze_memory:
            if sum(x.reward for x in trajectory) < 1:
                messages = self._build_messages(trajectory)
                messages[-1]["content"] += "\n\nThe task was not completed successfully. What should you do better next time? Be very concise. Respond with a single sentence."
                response = query_llm(messages)
                self.log("reflection", response)
                self.reflections.append(response)

    def _build_messages(self, trajectory: Trajectory) -> List[dict]:
        system_message = "You are playing a text-based game in which you must interact with your surroundings to complete a task.\n\n"
        system_message += trajectory.task_description
        system_message += "\n\nGiven the state, reflect on what has happened so far, explain your plan to accomplish the task and then output the next action to execute (use one of the action templates below)."
        system_message += "\n\nFor example:\nThe last action had the effect of... To accomplish the task, I will need to...\nCurrent subgoal: [subgoal]\nNext action: [action]"
        if trajectory[-1].action_prompt is not None:
            system_message += "\n\n" + trajectory[-1].action_prompt

        if len(self.reflections) > 0:
            system_message += "\n\nConsider the following tips:\n"
            system_message += "\n".join(self.reflections[-self.max_reflections:])

        messages = [dict(role="system", content=system_message)]

        state_strings = trajectory.build_string(use_task=False, use_actions=False, as_list=True, trim_state=self.trim_states)
        for i in range(len(trajectory)):
            state = trajectory[i]

            if i > 0:
                action_start = state.last_generation.lower().find("next action:")
                if action_start != -1:
                    action_start += len("next action:")
                    action_text = state.last_generation[:action_start+1] + " " + state.last_action
                else:
                    action_text = "To accomplish the task, I will need to {}. Next action: {}".format(state.last_action, state.last_action)
                messages.append(dict(role="assistant", content=action_text))

            prompt = state_strings[i]
            messages.append(dict(role="user", content=prompt))

        return messages

    def _act(self, trajectory: Trajectory) -> str:
        sub_trajectory = trajectory.slice(-self.max_history, None)
        messages = self._build_messages(sub_trajectory)
        response = query_llm(messages)
        self.log("action_messages", messages)
        self.log("action_generation", response)
        return response
