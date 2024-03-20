from sso.trajectory import Trajectory
from sso.llm import query_llm
from sso.agent import Agent
from sso.utils import clean_feature


class SkillsAgent(Agent):

    def act(self, trajectory: Trajectory) -> str:
        super().act(trajectory)
        return self._act(trajectory)

    def _update_memory(self, trajectory: Trajectory) -> None:
        self.memory.build(trajectory)

    def _act(self, trajectory: Trajectory) -> str:
        sub_trajectory = trajectory.slice(-self.max_history, None)

        system_message = "You are playing a text-based game in which you must interact with your surroundings to complete a task. You will occasionally be given posisible subgoals. You may choose to target one of these subgoals or ignore them.\n\n"
        system_message += sub_trajectory.task_description
        system_message += "\n\nGiven the state, reflect on what has happened so far, explain your plan to accomplish the task, output which of the given subgoals you are targeting next (match one of the subgoals in the prompt word for word or output \"none\"), and then output the next action to execute (use one of the action templates below)."
        system_message += "\n\nFor example:\nThe last action had the effect of... To accomplish the task, I will need to...\nCurrent subgoal: [subgoal]\nNext action: [action]"
        if sub_trajectory[-1].action_prompt is not None:
            system_message += "\n\n" + sub_trajectory[-1].action_prompt

        skills = self.memory.get_memories(trajectory=sub_trajectory)
        if len(skills) > 0:
            skill_text = "The following instructions contain potentially useful information about reaching subgoals:"
            for skill in skills:
                skill_text += "\n\nInstructions for reaching the subgoal \"{}\":\n\t".format(skill.target)
                skill_text += "\n\t".join("{}. {}".format(i + 1, instruction) for i, instruction in enumerate(skill.instructions))
            system_message += "\n\n" + skill_text

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
                    action_text = "To accomplish the task, I will need to {}.\nCurrent subgoal: none\nNext action: {}".format(state.last_action, state.last_action)
                messages.append(dict(role="assistant", content=action_text))

            prompt = state_strings[i]
            messages.append(dict(role="user", content=prompt))

        response = query_llm(messages)

        if "current subgoal:" in response.lower():
            current_skill = clean_feature(response.lower().split("current subgoal:")[-1].split("next action:")[0].strip())
            if current_skill != "none":
                for skill in reversed(skills):
                    if clean_feature(skill.target) == current_skill:
                        self.memory.log_sampled(len(trajectory) - 1, skill)
                        self.log("chosen_skill", skill.target)
                        trajectory[-1].skill_target = skill.target
                        break

        self.log("action_messages", messages)
        self.log("action_generation", response)
        return response
