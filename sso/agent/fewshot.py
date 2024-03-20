from sso.trajectory import Trajectory
from sso.llm import query_llm
from sso.agent import Agent


class FewshotAgent(Agent):

    def __init__(self, fewshot: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.fewshot = fewshot

    def act(self, trajectory: Trajectory) -> str:
        super().act(trajectory)

        # Get next action
        return self._act(trajectory)

    def _update_memory(self, trajectory: Trajectory) -> None:
        self.memory.insert(trajectory)

    def _act(self, trajectory: Trajectory) -> str:
        sub_trajectory = trajectory.slice(-self.max_history, None)

        system_message = "You are playing a text-based game in which you must interact with your surroundings to complete a task.\n\n"
        system_message += sub_trajectory.task_description
        system_message += "\n\nGiven the state, reflect on what has happened so far, explain your plan to accomplish the task and then output the next action to execute (use one of the action templates below)."
        system_message += "\n\nFor example:\nThe last action had the effect of... To accomplish the task, I will need to...\nCurrent subgoal: [subgoal]\nNext action: [action]"
        if sub_trajectory[-1].action_prompt is not None:
            system_message += "\n\n" + sub_trajectory[-1].action_prompt

        if len(self.memory.trajectories) > 0:
            examples = self.memory.get_memories(n=self.fewshot)
            system_message += "\n\nUse the following example trajector{} to help you accomplish the task:".format(
                "ies" if len(examples) > 1 else "y"
            )
            for traj in examples:
                system_message += "\n\n" + traj.build_string(trim_state=self.trim_states)

        messages = [dict(role="system", content=system_message)]

        state_strings = sub_trajectory.build_string(use_task=False, use_actions=False, as_list=True, trim_state=self.trim_states)
        for i in range(len(sub_trajectory)):
            state = sub_trajectory[i]

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

        response = query_llm(messages)
        self.log("action_messages", messages)
        self.log("action_generation", response)
        return response
