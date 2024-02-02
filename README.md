# Skill Set Optimization

## Installation

pip install python library requirements with:

```
pip install -r requirements.txt
```

## Usage

Executing the `main.py` file with the environment variable `OPENAI_API_KEY` will run SSO or baselines on ScienceWorld or NetHack tasks.

```
OPENAI_API_KEY=[your key] python main.py \
    --agent skills \
    --memory skills \
    --env scienceworld \
    --task measure-melting-point-known-substance \
    --train_variant_count 10 \
    --test_variants 399 354 390 418 335 409 377 410 385 367
```

By default, the script will train the agent for 30 iterations and then test its ability to transfer to the specified test variants.
Use `--test_init` and `--test_adapt` to also test the base agent's performance and the agent's ability to adapt to each of the test tasks in at most 5 iterations.

### Agent

Execute the SSO agent by using the below command line arguments:

```
--agent skills
--memory skills
```

Execute a Reflexion baseline using:

```
--agent reflexion
```

Execute a Fewshot baseline using:

```
--agent fewshot
--memory examples
```

### Environment

#### NetHack

Evaluate on our custom NetHack task using:

```
--env nethack
--task MiniHack-KeyLavaCross-v0
--max_history 5
--full_states
--test_iters 10
--train_iters 30
```

Other NetHack environments can be used by altering the `task` argument.
However, SSO has only been tested with `MiniHack-KeyLavaCross-v0`.
We recommend setting the `max_history` and `full_states` arguments for NetHack as shown above.

#### ScienceWorld

Evaluate on a ScienceWorld task and variants using:

```
--env scienceworld
--task measure-melting-point-known-substance
--train_variant_count 10
--test_variants 399 354 390 418 335 409 377 410 385 367
--test_iters 1
--train_iters 3
```

Note that `task` can be set to any valid ScienceWorld task, and variants can be selected randomly with `train_variant_count`/`test_variant_count` or by specifying specific variants with `train_variants`/`test_variants`.