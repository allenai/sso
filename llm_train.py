from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments
)
from sso.sft.dataloader import TrajectoryDataset
from transformers.integrations import TensorBoardCallback
from torch.utils.data import DataLoader
from argparse import ArgumentParser
HF_TOKEN = "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"


def train(model:str,
        load_path:str):
    tokenizer = AutoTokenizer.from_pretrained(model, token = HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model, token = HF_TOKEN)
    
    dataset = TrajectoryDataset(load_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # set arguments
    training_args = TrainingArguments(
        output_dir='./sft_results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./sft_logs',
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model("./sft_model")
    tokenizer.save_pretrained("./sft_model")

if __name__=="__main__":
    parser = ArgumentParser()

    # llama3 params
    parser.add_argument("--output", type=str, default="", help="output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="number of epochs to train llm")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="number of train batch size for one device")
    parser.add_argument("--save_steps", type=int, default=0, help="test frequency, if 0 will test once at end")
    parser.add_argument("--save_total_limit", action="store_true", help="test initial agent")
    parser.add_argument("--logging_dir", action="store_true", help="test adapting to test tasks")


    load_path = "/home/zhiyant/sso/chem/train"
    model = "meta-llama/Meta-Llama-3-8B-Instruct"


    train(model = model, load_path = load_path)

    # OOM 
    # try to use slurm instead of cpu
