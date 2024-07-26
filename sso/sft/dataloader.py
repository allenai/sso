import torch
from torch.utils.data import Dataset, DataLoader

from sso.trajectory import Trajectory
from sso.agent import Agent
import json
import os

class TrajectoryDataset(Dataset):
    def __init__(self, load_path, 
                tokenizer, 
                max_length=512, 
                trim_state=True):
        
        self.load_path = load_path
        self.trajectories: List[Trajectory] = []
        self.trim_state = trim_state

        self.tokenizer = tokenizer
        # self.pad_token = self.tokenizer.add_special_tokens("<pad>")

        self.load(self.load_path)
        self.max_length = max_length
        self.inputs = self._prepare_data()
    
    def load(self, load_path):
        for subfolder in os.listdir(load_path):
                subfolder_path = os.path.join(load_path, subfolder)

                if os.path.exists(os.path.join(subfolder_path, "trajectory.json")):
                    with open(os.path.join(subfolder_path, "trajectory.json"), "r") as f:
                        data = json.load(f)

                    trajectory = Trajectory.from_dict(data)
                    self.trajectories.append(trajectory)
    
    def _prepare_data(self):

        inputs = []

        for sub_trajectory in self.trajectories:
            state_strings = sub_trajectory.build_string(use_task=False, use_actions=False, as_list=True, trim_state=self.trim_state)
            messages = []
            
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
            
            text = " ".join([message['content'] for message in messages])
            encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
            inputs.append(encoded)
        
        return inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.inputs[idx]['input_ids'])
        attention_mask = torch.tensor(self.inputs[idx]['attention_mask'])
        labels = input_ids.clone()
        return 
        {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels
        }

