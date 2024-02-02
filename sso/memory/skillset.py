from typing import List, Union

import numpy as np
from tqdm import tqdm
from copy import deepcopy

from sso.trajectory import Trajectory
from sso.skill import Skill
from sso.memory import Memory


class SkillSetMemory(Memory):

    def __init__(
            self,
            max_trajectories: int = 10,
            num_beams: int = 20,
            max_skills: int = 5000,
            max_traj_len: int = 50,
            min_skill_len: int = 3,  # these numbers include the start and end states
            max_skill_len: int = 6,
            max_traj_count: int = 2,
            coverage_weight: float = .01,
            reward_weight: float = .01,
            state_weight: float = .1,
            action_weight: float = 1,
            sampled_weight: float = 1,
            **kwargs
        ):
        super().__init__(max_trajectories=max_trajectories, **kwargs)
        self.num_beams = num_beams
        self.max_traj_len = max_traj_len
        self.max_skills = max_skills
        self.min_skill_len = min_skill_len
        self.max_skill_len = max_skill_len
        self.max_traj_count = max_traj_count

        self.coverage_weight = coverage_weight
        self.reward_weight = reward_weight
        self.state_weight = state_weight
        self.action_weight = action_weight
        self.sampled_weight = sampled_weight

        self.last_skills = []
        self.built_trajectories = set()
        self.mean_coverage = np.mean(np.arange(self.min_skill_len, self.max_skill_len + 1))
        self.std_coverage = np.std(np.arange(self.min_skill_len, self.max_skill_len + 1))
        self.mean_reward = 0
        self.std_reward = 1
        self.all_state_similarities = []
        self.mean_state_similarity = 0
        self.std_state_similarity = 1
        self.all_action_similarities = []
        self.mean_action_similarity = 0
        self.std_action_similarity = 1

    def insert(self, trajectory: Trajectory) -> None:

        # Split trajectories into successful and unsuccessful
        inserted = None
        idx = 0
        while self.max_traj_len is not None and idx + self.max_traj_len < len(trajectory):
            sub_traj = trajectory.slice(idx, idx + self.max_traj_len)
            last_reward = [i for i, x in enumerate(sub_traj) if x.reward > 0]
            if len(last_reward) > 0 and last_reward[-1] >= self.min_skill_len:
                self.trajectories.append(sub_traj.slice(0, last_reward[-1] + 1))
                idx += last_reward[-1]
            idx += 1
        last_reward = [i for i, x in enumerate(trajectory) if x.reward > 0]
        if len(last_reward) > 0 and last_reward[-1] - idx >= self.min_skill_len:
            self.trajectories.append(trajectory.slice(idx, last_reward[-1] + 1))
            inserted = trajectory.slice(0, last_reward[-1] + 1)

        # Record skill feedback
        successful_cutoff = len(inserted) if inserted is not None else 0
        self.skill_feedback.append([])
        for t, skill in self.sampled:
            ret = sum(step.reward * self.discount ** i for i, step in enumerate(trajectory[t:]))
            self.skill_feedback[-1].append((skill, ret, t < successful_cutoff))
        while len(self.skill_feedback) > self.max_trajectories * self.skill_memory_scale:
            self.skill_feedback.pop(0)
        self.sampled = []

        if len(self.trajectories) < 2:
            return

        # Extract potential skills
        skills: List[Skill] = []
        for trajectory_idx, trajectory in enumerate(self.trajectories[-self.max_trajectories:]):
            if trajectory in self.built_trajectories:
                continue
            self.built_trajectories.add(trajectory)
            if trajectory_idx == 0:
                continue
            trajectory_idx += max(0, len(self.trajectories) - self.max_trajectories)
            for start_idx in range(len(trajectory)):
                for end_idx in range(start_idx + self.min_skill_len, start_idx + self.max_skill_len):
                    if end_idx > len(trajectory) + 1:
                        break

                    skill = Skill(max_trajectories=self.max_traj_count)
                    skill.add(trajectory.slice(start_idx, end_idx), trajectory_idx, start_idx, end_idx)

                    for i, traj1 in enumerate(self.trajectories[-self.max_trajectories:trajectory_idx]):
                        i += max(0, len(self.trajectories) - self.max_trajectories)
                        proposed_skill = self.add_to_skill(skill, traj1, i)
                        if proposed_skill is not None:
                            skills.append(proposed_skill)
                            self.all_state_similarities.append(proposed_skill.state_similarity)
                            self.all_action_similarities.append(proposed_skill.action_similarity)

        if len(skills) == 0:
            print("Extracted no skills.")
            return

        # Set similarity normalization
        self.mean_state_similarity = np.mean(self.all_state_similarities)
        self.std_state_similarity = np.std(self.all_state_similarities)
        self.mean_action_similarity = np.mean(self.all_action_similarities)
        self.std_action_similarity = np.std(self.all_action_similarities)
        skills = [x for x in skills if x.state_similarity > self.mean_state_similarity and x.action_similarity > self.mean_action_similarity]

        # Set reward normalization
        rewards = [x.reward for traj in self.trajectories[-self.max_trajectories:] for x in traj if x.reward > 0]
        self.mean_reward = np.mean(rewards)
        self.std_reward = np.std(rewards)

        # Sort skills
        self.last_skills = list(set(self.last_skills + skills))
        self.last_skills = sorted(self.last_skills, key=lambda x: self.score_skill(x), reverse=True)

        # Only keep skills used in beam search
        self.memory, sampled_skills = self.beam_search(self.last_skills, return_sampled=True)
        self.last_skills = [skill for skill in self.last_skills if skill in sampled_skills]
        self.last_skills = sorted(self.last_skills, key=lambda x: sampled_skills[x], reverse=True)[:self.max_skills]

    def build(self, trajectories: Union[Trajectory, List[Trajectory]] = [], resample: bool = False) -> None:
        super().build(trajectories)

        if resample:
            self.memory = self.beam_search(self.last_skills, return_sampled=False)

        print("Building selected skill set...")
        proposed_skills = self.memory.copy()
        self.memory = []
        for skill in tqdm(proposed_skills):
            skill.build()
            if not self.check_similar_skills(skill):
                self.memory.append(skill)

    def _norm_score(self, score: float, mean: float, std: float) -> float:
        return max(0, (score - mean + 2 * std) / std)

    def score_skill(self, skill: Skill) -> float:
        score = 0
        if self.coverage_weight > 0:
            score += self.coverage_weight * self._norm_score(skill.step_count(), self.mean_coverage, self.std_coverage)
        if self.reward_weight > 0:
            score += self.reward_weight * self._norm_score(skill.reward(), self.mean_reward, self.std_reward)
        if self.state_weight > 0:
            score += self.state_weight * self._norm_score(skill.state_similarity, self.mean_state_similarity, self.std_state_similarity)
        if self.action_weight > 0:
            score += self.action_weight * self._norm_score(skill.action_similarity, self.mean_action_similarity, self.std_action_similarity)
        if self.sampled_weight > 0:
            for episode in self.skill_feedback:
                for skl, ret, suc in episode:
                    if skl == skill:
                        score += (1 if suc else -1) * self.sampled_weight * self._norm_score(ret, self.mean_reward, self.std_reward)
        score /= (self.coverage_weight + self.reward_weight + self.state_weight + self.action_weight + self.sampled_weight)
        return score

    def add_to_skill(self, skill: Skill, trajectory: Trajectory, traj_idx: int) -> Skill:
        best_skill = None
        for t1 in range(len(trajectory)):
            for t2 in range(t1 + skill.traj_len(),
                            t1 + skill.traj_len() + 1):
                if t2 <= len(trajectory) and t2 - t1 >= self.min_skill_len:
                    new_skill = deepcopy(skill)
                    if new_skill.try_add(trajectory.slice(t1, t2), traj_idx, t1, t2):
                        if best_skill is None or \
                                new_skill.state_similarity * self.state_weight + new_skill.action_similarity * self.action_weight > \
                                best_skill.state_similarity * self.state_weight + best_skill.action_similarity * self.action_weight:
                            best_skill = new_skill
        return best_skill

    def beam_search(self, skills: List[Skill], return_sampled: bool = False) -> List[Skill]:
        beams = [dict(skillset=[], unused=skills.copy()) for _ in range(self.num_beams)]
        sampled_skills = dict()
        seen = set()
        while any(len(b["unused"]) > 0 for b in beams):
            new_beams = [dict(skillset=b["skillset"], unused=[]) for b in beams]
            for beam in beams:
                for _ in range(self.num_beams):
                    unused = beam["unused"].copy()
                    idx = 0
                    while idx < len(unused):
                        if not all(x.is_compatible(unused[idx]) for x in beam["skillset"]):
                            unused.pop(idx)
                        elif tuple(beam["skillset"] + [unused[idx]]) in seen:
                            idx += 1
                        else:
                            new_skill = unused.pop(idx)
                            if new_skill not in sampled_skills:
                                sampled_skills[new_skill] = 0
                            sampled_skills[new_skill] += 1
                            new_beams.append(dict(skillset=beam["skillset"] + [new_skill], unused=unused))
                            seen.add(tuple(new_beams[-1]["skillset"]))
                            break
            beams = sorted(
                new_beams,
                key=lambda x: sum(self.score_skill(skill) for skill in x["skillset"]),
                reverse=True
            )[:self.num_beams]

        if return_sampled:
            return beams[0]["skillset"], sampled_skills
        else:
            return beams[0]["skillset"]