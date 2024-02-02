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
from sso.agent.skills import IncontextAgent
from sso.agent.fewshot import FewshotAgent
from sso.agent.reflexion import ReflexionAgent
from sso.memory.skillset import SkillSetMemory
from sso.memory.examples import ExamplesMemory
from sso.trajectory import Trajectory
from sso.llm import set_default_model


def run_episode(
    env: Env,
    agent: Agent,
    task_id: str = None,
    test: bool = False,
) -> Tuple[str, Trajectory, bool, float]:

    done = False
    state, info = env.reset(task_id=task_id, test=test)
    trajectory = Trajectory(task_description=info["task_description"])
    trajectory.insert(state)

    agent.reset_log()
    agent.log("task_id", info["task_id"])
    agent.log("task_description", info["task_description"])

    pbar = tqdm(total=env.max_steps)
    score = 0
    while not done:
        action = agent.act(trajectory)
        state, reward, done, info = env.step(action)
        if reward > 0:
            score += reward
        trajectory.insert(state)
        pbar.update(1)
    pbar.close()

    agent.record_done(trajectory)
    agent.log("trajectory_success", info["success"])
    agent.log("score", score)
    agent.log("length", len(trajectory))

    return info["task_id"], trajectory, info["success"], score


def log_results(
    agent: Agent,
    trajectory: Trajectory,
    save_path: str,
    iteration: int,
    task_id: str,
    success: bool,
    score: float
):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "logs.json"), "w") as f:
        json.dump(agent.get_log(), f, indent=4)
    with open(os.path.join(save_path, "trajectory.json"), "w") as f:
        json.dump(trajectory.to_dict(), f, indent=4)
    agent.save(save_path)
    print("Iter: {}, task: {}, success: {}, score: {}".format(iteration, task_id, success, score))


def run_experiment(
    env: Env,
    agent: Agent,
    results_dir: str,
    experiment_name: str,
    iteration: int,
    temp: float = 1,
    train: bool = True,
    use_test_tasks: bool = False
):
    agent.train(train)
    set_default_model(temp=temp)
    task_id, trajectory, success, score = run_episode(env, agent, test=use_test_tasks)
    save_path = os.path.join(results_dir, experiment_name, str(iteration))
    log_results(agent, trajectory, save_path, iteration, task_id, success, score)
    return score, success


def run(
    agent: Agent,
    env: Env,
    results_dir: str = "results",
    train_count: int = 5,
    adapt_count: int = 5,
    test_count: int = 1,
    test_freq: int = 0,
    train_temp: float = 1,
    test_temp: float = 0,
    test_init: bool = False,
    test_adapt: bool = False
) -> Tuple[Dict[str, List[bool]], Dict[str, List[bool]]]:

    def get_scores(experiment_name):
        s = []
        for exp in os.listdir(os.path.join(results_dir, experiment_name)):
            with open(os.path.join(results_dir, experiment_name, exp, "logs.json"), "r") as f:
                s.append(json.load(f)[-1]["score"])
        return s
    results = dict()

    # Test base agent
    if test_init and (train_count == 0 or train_count > 0):
        print("\n##############################\nTesting base agent")
        for test_iter in range(test_count * env.num_test):
            run_experiment(env, agent, results_dir, "test_init", test_iter, temp=test_temp, train=False, use_test_tasks=True)
        if test_count > 0:
            results["base"] = np.mean(get_scores("test_init"))
            with open(os.path.join(results_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)

    # Adapt
    if test_adapt:
        print("\n##############################\nTesting base adaptation")
        for test_id in env.test_ids:
            results["adapt_" + test_id] = []
            agent.clear()
            for train_iter in range(adapt_count):
                score, _ = run_experiment(env, agent, results_dir, "test_adapt/{}/{}".format(test_id, train_iter),
                                          "train", temp=test_temp, train=True, use_test_tasks=True)
                results["adapt_" + test_id].append(score)
        results["adapt_best"] = np.mean([np.max(results["adapt_" + test_id]) for test_id in env.test_ids])
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        agent.clear()

    # Train agent
    print("\n##############################\nTraining agent")
    train_scores = []
    for train_iter in range(train_count * env.num_train):
        score = run_experiment(env, agent, results_dir, "train", train_iter, temp=train_temp)
        train_scores.append(score)
        if test_freq > 0 and (train_iter + 1) % test_freq == 0 and train_iter < (train_count * env.num_train) - 1:
            for test_iter in range(test_count * env.num_test):
                run_experiment(env, agent, results_dir, "test_iter{}".format(train_iter), test_iter, temp=test_temp,
                               train=False, use_test_tasks=True)
                results["test_iter{}".format(train_iter)] = np.mean(get_scores("test_iter{}".format(train_iter)))
                with open(os.path.join(results_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=4)

    # Transfer
    print("\n##############################\nTesting transfer agent")
    if train_count * env.num_train > env.num_test or not os.path.exists(os.path.join(results_dir, "test_transfer")):
        for test_iter in range(test_count * env.num_test):
            run_experiment(env, agent, results_dir, "test_transfer", test_iter, temp=test_temp, train=False, use_test_tasks=True)
    if test_count > 0:
        results["transfer"] = np.mean(get_scores("test_transfer"))
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':

    parser = ArgumentParser()

    # Experiment params
    parser.add_argument("--output", type=str, default="results", help="output directory")
    parser.add_argument("--train_iters", type=int, default=3, help="number of iterations to run")
    parser.add_argument("--test_iters", type=int, default=1, help="number of test iterations to run")
    parser.add_argument("--test_freq", type=int, default=0, help="test frequency, if 0 will test once at end")
    parser.add_argument("--test_init", action="store_true", help="test initial agent")
    parser.add_argument("--test_adapt", action="store_true", help="test adapting to test tasks")

    # Agent params
    parser.add_argument("--agent", type=str, default="skills", help="agent type")
    parser.add_argument("--load", type=str, default=None, help="directory to load agent from")
    parser.add_argument("--model", type=str, default="gpt-4-0613", help="model name")
    parser.add_argument("--train_temp", type=float, default=0.7, help="Generation temperature for the llm during training")
    parser.add_argument("--test_temp", type=float, default=0, help="Generation temperature for the llm during testing")
    parser.add_argument("--max_history", type=int, default=10, help="number of past steps to keep in history")
    parser.add_argument("--full_states", action="store_true", help="do not trim states to only keep new information")
    parser.add_argument("--similarity_metric", type=str, default="text-embedding-ada-002", help="similarity metric to use, iou or model name")

    # Env params
    parser.add_argument("--env", type=str, default="nethack", help="environment type")
    parser.add_argument("--task", type=str, default="MiniHack-KeyLavaCross-v0", help="task to run")
    parser.add_argument("--train_variant_count", type=int, default=10, help="number of variants to train on")
    parser.add_argument("--test_variant_count", type=int, default=10, help="number of variants to test on")
    parser.add_argument("--train_variants", type=int, nargs="+", default=None, help="train task variants, if None will use default task split")
    parser.add_argument("--test_variants", type=int, nargs="+", default=None, help="test task variants, if None will use default task split")

    # Memory params
    parser.add_argument("--memory", type=str, default="skills", help="memory type")
    parser.add_argument("--reward_weight", type=float, default=0.1, help="weight for trajectory reward in skill score")
    parser.add_argument("--state_weight", type=float, default=1, help="weight for state similarity in skill score")
    parser.add_argument("--action_weight", type=float, default=1, help="weight for action similarity in skill score")
    parser.add_argument("--coverage_weight", type=float, default=0.01, help="weight for task coverage in skill score")

    args = parser.parse_args()

    # Set LLMs
    set_default_model(model=args.model, temp=args.train_temp,
                      embedding=None if args.similarity_metric == "iou" else args.similarity_metric)

    # Set memory
    if args.memory == "skills":
        memory = SkillSetMemory(
            reward_weight=args.reward_weight,
            state_weight=args.state_weight,
            action_weight=args.action_weight,
            coverage_weight=args.coverage_weight,
        )
    elif args.memory == "examples":
        memory = ExamplesMemory()
    else:
        raise ValueError(f"Unknown memory type: {args.memory}")

    # Set agent
    agent_args = dict(
        memory=memory,
        max_history=args.max_history,
        trim_states=not args.full_states
    )
    if args.agent == "skills":
        agent = IncontextAgent(**agent_args)
    elif args.agent == "fewshot":
        agent = FewshotAgent(**agent_args)
    elif args.agent == "reflexion":
        agent = ReflexionAgent(**agent_args)
    else:
        raise ValueError("Invalid agent: {}".format(args.agent))
    if args.load is not None:
        agent.load(args.load)

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
    elif args.env == "nethack":
        env = NetHackTask(**env_args)
    else:
        raise ValueError("Invalid env: {}".format(args.env))

    # Run
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    run(agent, env, results_dir=args.output, train_count=args.train_iters, test_count=args.test_iters,
        test_freq=args.test_freq, train_temp=args.train_temp, test_temp=args.test_temp,
        test_init=args.test_init, test_adapt=args.test_adapt)
