from typing import List, Tuple, Dict, Any
import random
from scienceworld import ScienceWorldEnv

from sso.env import Env
from sso.trajectory import State
from sso.utils import get_similar_action


class ScienceWorld(Env):

    ACTION_TEMPLATES = [
        "read OBJ",
        "activate OBJ",
        "deactivate OBJ",
        "open OBJ",
        "close OBJ",
        "pick up OBJ",
        "look in OBJ",
        "focus on OBJ",
        "move OBJ to CONTAINER",
        "pour OBJ in CONTAINER",
        "mix CONTAINER",
        "teleport LOCATION",
        "go LOCATION",
        "wait"
    ]

    def __init__(
            self,
            task: str = None,
            train_variant_count: int = 10,
            test_variant_count: int = 10,
            train_variants: List[int] = None,
            test_variants: List[int] = None,
            max_repeat: int = 10,
            seed: int = 42
        ):
        self.env = ScienceWorldEnv()
        random.seed(seed)
        assert task is not None, "You must specify a task to load"
        self.env.load(task)

        self.train_variants = train_variants
        if self.train_variants is None:
            self.train_variants = [
                x for x in self.env.getVariationsTrain()
                if test_variants is None or x not in test_variants
            ]
            self.train_variants = random.sample(self.train_variants, min(len(self.train_variants), train_variant_count))            

        self.test_variants = test_variants
        if self.test_variants is None:
            self.test_variants = [
                x for x in self.env.getVariationsTest()
                if train_variants is None or x not in train_variants
            ]
            self.test_variants = random.sample(self.test_variants, min(len(self.test_variants), test_variant_count))

        self.max_repeat = max_repeat
        self.task = task

        self.last_actions = []
        self.t = 0
        self._max_steps = None
        self.variant = None
        self.next_test_idx = 0
        self.next_train_idx = 0
        self.last_focus = None
        self.last_obs = None

    def reset(self, task_id: str = None, test: bool = False, **kwargs) -> Tuple[State, Dict[str, Any]]:
        if task_id:
            self.variant = task_id.split("_")[1]
            self.variant = int(self.variant)
        elif test:
            self.variant = self.test_variants[self.next_test_idx]
            self.next_test_idx += 1
            if self.next_test_idx >= len(self.test_variants):
                self.next_test_idx = 0
        else:
            self.variant = self.train_variants[self.next_train_idx]
            self.next_train_idx += 1
            if self.next_train_idx >= len(self.train_variants):
                self.next_train_idx = 0
        self.env.load(self.task, self.variant, "easy", generateGoldPath=True)

        self.last_obs, info = self.env.reset(**kwargs)
        self.last_actions = []
        self.t = 0
        self._max_steps = int(len(self.env.getGoldActionSequence()) * 1.5)
        self.last_focus = None
        info["invalid"] = "no known action matches that input" in self.last_obs.lower()
        info["obs"] = self.last_obs
        info["task_id"] = "{}_{}".format(self.task, self.variant)
        info["task_description"] = info["taskDesc"]
        info["success"] = False
        state = self.get_state(info, None, None, 0, False)
        return state, info

    def step(self, action: str, **kwargs) -> Tuple[State, float, bool, Dict[str, Any]]:
        parsed_action = self.parse_action(action)
        self.last_obs, reward, done, info = self.env.step(parsed_action, **kwargs)
        if "focus" in self.last_obs.lower():
            self.last_focus = self.last_obs
        self.last_actions.append(parsed_action)
        info["obs"] = self.last_obs
        info["task_id"] = "{}_{}".format(self.task, self.variant)
        info["task_description"] = info["taskDesc"]
        info["success"] = info["score"] >= 100
        info["last_raw_action"] = action
        self.t += 1
        if self.t >= self.max_steps or len(self.last_actions) > self.max_repeat and \
                all(x == self.last_actions[-1] for x in self.last_actions[-self.max_repeat:]):
            done = True
        reward /= 100
        state = self.get_state(info, action, parsed_action, reward, done)
        return state, reward, done, info

    def parse_action(self, action: str) -> str:
        action = action.lower()
        if "next action:" in action:
            action = action.split("next action:")[-1].strip().split("\n")[0].strip()
        parsed_action = None
        if "ambiguous request" in self.last_obs.lower():
            nums = [c for c in action if c.isnumeric()]
            if len(nums) > 0:
                parsed_action = nums[0]
        if parsed_action is None:
            parsed_action = get_similar_action(action, self.env.getValidActionObjectCombinations())
        if parsed_action is None:
            parsed_action = action
        return parsed_action

    def get_action_prompt(self):
        res = "Generate the action using one of the following templates, where OBJ is an object in the scene and LOCATION is an adjacent room and CONTAINER is an object that can contain or hold other objects such as a pot, table, or inventory."
        res += " The \"focus on OBJ\" action is extremely critical and should not be used to look at or shift your attention to a specific object. It should only be used as described in the task description. Using this action inappropriately will result in task failure."
        res += " The \"wait\" action is used to wait for time to pass during tasks that require the passage of time."
        res += "\nTemplates:\n\t" + "\n\t".join(self.ACTION_TEMPLATES)
        return res

    def get_state(self, state: Dict[str, Any], last_generation: str, last_action: str, reward: float, done: bool) -> State:
        return State(
            observation=state["obs"][:state["obs"].index("In it, you see")].strip() if "In it, you see" in state["obs"] else state["obs"],
            state_features=self.get_features(state, self.last_focus),
            state_description=self.get_description(state),
            last_generation=last_generation,
            last_action=last_action,
            reward=reward,
            done=done,
            task_description=state["taskDesc"],
            action_prompt=self.get_action_prompt(),
            all_templates=self.ACTION_TEMPLATES,
        )

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def num_train(self) -> int:
        return len(self.train_variants)

    @property
    def num_test(self) -> int:
        return len(self.test_variants)

    @property
    def train_ids(self) -> Tuple[str]:
        return tuple(["{}_{}".format(self.task, v) for v in self.train_variants])

    @property
    def test_ids(self) -> Tuple[str]:
        return tuple(["{}_{}".format(self.task, v) for v in self.test_variants])

    @staticmethod
    def parse_commas(sub_feats: str) -> List[str]:
        p = []
        idx = 0
        next_feat = ""
        while idx < len(sub_feats):
            if sub_feats[idx] == ",":
                p.append(next_feat)
                next_feat = ""
                idx += 1
            elif sub_feats[idx:].startswith("(containing") and ")" in sub_feats[idx:]:
                p += ScienceWorld.parse_commas(
                    sub_feats[idx + len("(containing") : idx + sub_feats[idx:].index(")")]
                )
                paren_size = sub_feats[idx:].index(")") + 1
                next_feat += sub_feats[idx:idx+paren_size]
                p.append(next_feat)
                next_feat = ""
                idx += paren_size
            else:
                next_feat += sub_feats[idx]
                idx += 1
        p.append(next_feat)
        return p

    @staticmethod
    def parse_feature(feature: str) -> List[str]:
        parsed = [feature]
        if ":" in feature and ": which is" not in feature:
            parsed += ScienceWorld.parse_commas(feature[feature.index(":") + 1:])
        elif "(containing" in feature:
            parsed += ScienceWorld.parse_commas(feature[feature.index("(containing") + len("(containing") : feature.index(")")])
        return [p.strip(" \n\t.,").lower() for p in parsed if p.strip(" \n\t.,") not in ["", "nothing"]]

    @staticmethod
    def get_description(state: Dict[str, Any]) -> str:
        res = state["obs"]
        if "you see" not in res.lower():
            res += "\n" + state["look"]
        if state["inv"] not in res:
            res += "\n" + state["inv"]
        return res

    @staticmethod
    def get_features(state: Dict[str, Any], last_focus=None) -> List[str]:
        features = []
        if last_focus is not None:
            features.append(last_focus)

        for feat in state["look"].split("\n"):
            if feat.strip() != "" and feat != "You also see:":
                if feat.startswith("\t"): 
                    features.extend([x for x in ScienceWorld.parse_feature(feat)])
                else:
                    features.append(feat.replace("In it, you see:", "").strip().lower())

        for feat in state["inv"].split("\n"):
            if feat.strip() != "" and feat != "In your inventory, you see:":
                features.extend([x + " in your inventory" for x in ScienceWorld.parse_feature(feat)])

        return [x.replace("\n", ", ").replace("\t", " ") for x in features]
