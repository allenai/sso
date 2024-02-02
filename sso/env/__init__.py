from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod, abstractproperty

from sso.trajectory import State


class Env(ABC):

    @abstractproperty
    def max_steps(self) -> int:
        """
        :return: maximum number of steps in the simulation
        """
        raise NotImplementedError

    @abstractproperty
    def num_train(self) -> int:
        """
        :return: number of training variants
        """
        raise NotImplementedError

    @abstractproperty
    def num_test(self) -> int:
        """
        :return: number of test variants
        """
        raise NotImplementedError

    @abstractproperty
    def train_ids(self) -> Tuple[str]:
        """
        :return: training task IDs
        """
        raise NotImplementedError

    @abstractproperty
    def test_ids(self) -> Tuple[str]:
        """
        :return: test task IDs
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, task_id: str = None, test: bool = False, **kwargs) -> Tuple[State, Dict[str, Any]]:
        """
        Reset the simulation.

        :param test: whether to use a test task
        :param kwargs: additional arguments
        :return: initial state and info
        """
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action: str) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Perform an action in the simulation.

        :param action: action to perform
        :return: next state, reward, done, info
        """
        raise NotImplementedError
