from sso.trajectory import Trajectory
import json
from typing import List, Dict
import os
from sso.llm import get_response_llama3, get_embedding_llama3

from typing import Tuple, Dict, List
from argparse import ArgumentParser
from tqdm import tqdm
import os
import json
import numpy as np

from sso.env import Env
from sso.env.scienceworld import ScienceWorld
from sso.env.nethack.base import NetHackTask
from sso.agent import Agent
from sso.agent.skills import SkillsAgent
from sso.agent.fewshot import FewshotAgent
from sso.agent.reflexion import ReflexionAgent
from sso.memory.skillset import SkillSetMemory
from sso.memory.examples import ExamplesMemory
from sso.memory.subtrajset import MemoryBasedonRewards

from sso.trajectory import Trajectory

model = "meta-llama/Meta-Llama-3-8B"


# load an expert trajectory from the file
load_path = "test_trajectory"
with open(os.path.join(load_path, "trajectory.json"), "r") as f:
    data = json.load(f)
trajectory = Trajectory.from_dict(data)


def test_act(trajectory, valid_action):
    message = ""
    
    message += trajectory[-3].state_description
    message += "\n"
    message += "You choose to:" + trajectory[-2].last_action

    
    final_output = get_response_llama3(model, message, valid_action)
    print(final_output)

def test_run_episode(env):

    done = False
    
    state, info, valid_action = env.reset(test=True)
    
    test_act(trajectory, valid_action)

    # test_act(trajectory, valid_action)
    # pbar = tqdm(total = env.max_steps)
    # score = 0

    # while not done:
    #     action = test_act(trajectory, valid_action)
    #     state, reward, done, info = env.step(action)

    #     if reward > 0:
    #         score += reward
    #     trajectory.insert(state)
    #     pbar.update(1)
    # pbar.close()

    # print("trajectory_success", info["success"])
    # print("score", score)
    # print("length", len(trajectory))

    # return info["task_id"], trajectory, info["success"], score


if __name__ == '__main__':

    parser = ArgumentParser()

    # Experiment params
    parser.add_argument("--output", type=str, default="results_chem_test", help="output directory")
    parser.add_argument("--train_iters", type=int, default=3, help="number of iterations to run")
    parser.add_argument("--test_iters", type=int, default=1, help="number of test iterations to run")
    parser.add_argument("--test_freq", type=int, default=0, help="test frequency, if 0 will test once at end")
    parser.add_argument("--test_init", action="store_true", help="test initial agent")
    parser.add_argument("--test_adapt", action="store_true", help="test adapting to test tasks")

    # Env params
    parser.add_argument("--env", type=str, default="scienceworld", help="environment type")
    parser.add_argument("--task", type=str, default="chemistry-mix", help="task to run")
    parser.add_argument("--train_variant_count", type=int, default=10, help="number of variants to train on")
    parser.add_argument("--test_variant_count", type=int, default=10, help="number of variants to test on")
    parser.add_argument("--train_variants", type=int, nargs="+", default=None, help="train task variants, if None will use default task split")
    parser.add_argument("--test_variants", type=int, nargs="+", default=None, help="test task variants, if None will use default task split")

    args = parser.parse_args()

    # Set env
    env_args = dict(
        task=args.task,
        train_variant_count=args.train_variant_count,
        test_variant_count=args.test_variant_count,
        train_variants=args.train_variants,
        test_variants=args.test_variants
    )

    if args.env == "scienceworld":
        env = ScienceWorld(**env_args)

    test_run_episode(env)