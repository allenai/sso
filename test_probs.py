from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# test to find probability tensor containing 'inf', 'nan' or element < 0
import torch


logits = torch.tensor([[2.0, 2.0, 3.0],
                        [4.0, 1.0, 6.0],
                        [7.0, 0.0, 9.0]])

constrained_logits = logits.clone().fill_(-float('inf'))
idx = [[1],[0]]
mask = torch.zeros_like(logits, dtype=torch.bool)
mask[:, idx] = True
constrained_logits[mask] = logits[mask]

print("constrained_logits:",constrained_logits)

probs = torch.nn.functional.softmax(constrained_logits, dim=-1)


if torch.any(torch.isinf(probs)) or torch.any(torch.isnan(probs)) or torch.any(probs < 0):
    print("Warning: probs contains 'inf', 'nan', or element < 0.")


inf_nan_indices = torch.nonzero(torch.isinf(probs) | torch.isnan(probs) | (probs < 0), as_tuple=False)

print("Indices of 'inf', 'nan', or element < 0 in probs:")
print(inf_nan_indices)


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = self._prepare_data(trajectories)

    def _prepare_data(self, trajectories):
        inputs = []
        for trajectory in trajectories:
            text = " ".join(trajectory)
            encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length)
            inputs.append(encoded)
        return inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.inputs[idx]['input_ids'])
        attention_mask = torch.tensor(self.inputs[idx]['attention_mask'])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 假设你的trajectory对象是一个包含文本state的list
trajectories = [
    ["state1 of trajectory1", "state2 of trajectory1"],
    ["state1 of trajectory2", "state2 of trajectory2"],
    # 继续添加更多的trajectory
]

# 加载tokenizer和模型
tokenizer = LlamaTokenizer.from_pretrained('huggingface/llama-3')
model = LlamaForCausalLM.from_pretrained('huggingface/llama-3')

# 创建数据集和dataloader
dataset = TrajectoryDataset(trajectories, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()
