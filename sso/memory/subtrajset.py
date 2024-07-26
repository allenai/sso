from typing import List, Union, Tuple
import os
import json

import numpy as np
from tqdm import tqdm
from copy import deepcopy

from sso.trajectory import Trajectory
from sso.memory import Memory

from sso.llm import query_llm
from sso.utils import get_state_similarity


class MemoryBasedonRewards(Memory):

    def __init__(self, 
                 max_traj_len: int = 4,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.max_traj_len = max_traj_len
        self.gamma = 0.9

        self.best_reward = 0

        self.return_weight = 0.1
        self.idx_weight = 0.3
        self.similarity_weight = 0.9
        self.state_count_weight = -0.2

        self.mean_ret = 0
        self.std_ret = 0
        self.mean_similarity = 0
        self.std_similarity = 0

        self.state_visits = dict()


        # load an expert trajectory into the memory


        # templates the 
        
    
    def save(self, save_path: str) -> None:

        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, "trajectories.json"), "w") as f:
            json.dump(dict(
                trajectories=[s.to_dict() for s in self.trajectories],
            ), f, indent=4)

    def load_and_reboot(self, load_path:str, is_single:bool=True):

        if is_single:
            with open(os.path.join(load_path, "trajectory.json"), "r") as f:
                data = json.load(f)
            trajectory = Trajectory.from_dict(data)
            self.build(trajectory)
        
        else:
            for subfolder in os.listdir(load_path):
                subfolder_path = os.path.join(load_path, subfolder)
                if os.path.exists(os.path.join(subfolder_path, "trajectory.json")):
                    with open(os.path.join(subfolder_path, "trajectory.json"), "r") as f:
                        data = json.load(f)

                    trajectory = Trajectory.from_dict(data)
                    self.build(trajectory)
        # append the whole trajectory 
        

    def get_discounted_rewards(self, trajectory: Trajectory) -> List[float]:
        discounted_reward = []
        _add = 0
        for state in reversed(trajectory):
            if state.reward is not None:
                _add = state.reward + self.gamma * _add
                discounted_reward.insert(0, _add)
        return discounted_reward

    def insert(self, trajectory: Trajectory) -> None:
        # print the current state
        print("###################### Current Trajectory ######################")
        for state in trajectory:
            print(state.last_action, state.reward)
            if state in self.state_visits:
                self.state_visits[state] += 1
            else:
                self.state_visits[state] = 1
        
        # remove trailing zeros
        reward = [x.reward for x in trajectory]
        while reward and reward[-1] == 0:
            reward.pop()
        
        # caculate the discounted rewards of trajectory
        discounted_rewards = self.get_discounted_rewards(trajectory)

        discounted_rewards = discounted_rewards[:len(reward)]

        sum_discounted_rewards = []
        # fixed length of sub_traj
        length = self.max_traj_len

        while True:
            
            best_return = -np.inf
            sidx = None

         
            for idx in range(len(discounted_rewards) - length + 1):
                
                # cumm_return = best_return - discounted_rewards[idx - 1] + discounted_rewards[idx + length - 1]
                cumm_return = sum(discounted_rewards[idx : idx + length + 1])
                if cumm_return > best_return:
                    best_return = cumm_return
                    sidx = idx
            
            if sidx is None or sidx + length - 1 > len(discounted_rewards):
                break
            else:
                sub_traj = trajectory.slice(sidx, sidx + length) # best_sub_traj
                sum_discounted_rewards.append(best_return)
                self.trajectories.append(sub_traj)
                for idx in range(sidx, sidx + length):
                    discounted_rewards[idx] = -(np.inf)

                    

        self.mean_ret = np.mean(sum_discounted_rewards)
        self.std_ret = np.std(sum_discounted_rewards)


        print("###################### Selected Subtrajectories in insert() ######################")
        for i, subtraj in enumerate(self.trajectories):
            print(f"###################### subtrajectory {i} ######################")
            for x in subtraj:
                print(x.last_action, x.reward)

        return

    def norm_score(self, score, mean, std):
        return max(0, (score - mean + 2 * std) / std)

    def get_memories(self, trajectory: Trajectory = None, n: int = None) -> List[Trajectory]:

        print("###################### Current Trajectory ######################")
        for state in trajectory:
            print(state.last_action, state.reward)


        if n is None or n > len(self.trajectories):
            n = len(self.trajectories)
        
        combined_list = []
        all_state_similarity = []
        for i in range(len(self.trajectories)):

            ret = sum(self.get_discounted_rewards(self.trajectories[i])) 
            
            idx = (i / len(self.trajectories)) 

            state_similarities = get_state_similarity(self.trajectories[i][0], trajectory[-1]) if trajectory is not None else 0
            
            state_count = sum(self.state_visits[state] for state in self.trajectories[i])

            all_state_similarity.append(state_similarities)

            combined_list.append([ret, idx, state_similarities, state_count, self.trajectories[i]])

        self.mean_similarity = np.mean(all_state_similarity)
        self.std_similarity = np.std(all_state_similarity)

        
        for i, item in enumerate(combined_list):
            item[0] = self.norm_score(item[0], self.mean_ret, self.std_ret) * self.return_weight
            item[1] = item[1] * self.idx_weight
            item[2] = self.norm_score(item[2], self.mean_similarity, self.std_similarity) * self.similarity_weight
            item[3] = item[3] * self.state_count_weight
            
           
        sorted_combined_list = sorted(combined_list, 
                                      key = lambda x:x[2]+x[0]+x[1]+x[3],
                                      reverse=True) # desc

        sorted_trajectories = [item[4] for item in sorted_combined_list[:n]]

        # todo save the log into txt files
        print("###################### Result for get_memories() ######################")
        for i, sub_traj in enumerate(sorted_trajectories):
            print(f"###################### subtrajectory {i} ######################")
            for x in sub_traj:
                print(x.last_action, x.reward)


        return sorted_trajectories

    

