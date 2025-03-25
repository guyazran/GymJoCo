from typing import SupportsFloat

import numpy as np
from gymnasium import ObservationWrapper, RewardWrapper
from gymnasium.core import ObsType, WrapperObsType
from gymnasium.spaces import Box

from gymjoco.common.metrics import position_euclidean_distance


class BallworldRLObsWrapper(ObservationWrapper):
    def observation(self, observation: ObsType) -> WrapperObsType:
        # NOTE: this is a hacky way to add the goal position to the observation space
        self._observation_space = Box(-np.inf, np.inf, shape=(self.unwrapped.observation_space.shape[0] + 3,))

        goal_pos = self.unwrapped.task.obj_poses['ball/']['goal_com']
        return np.concatenate((observation, goal_pos))


class BallworldRLRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.__prev_dist = None

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        goal_pos = self.unwrapped.task.obj_poses['ball/']['goal_com']
        self.__prev_dist = position_euclidean_distance(self.unwrapped.sim.data.qpos[:3], goal_pos)
        return out

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        goal_pos = self.unwrapped.task.obj_poses['ball/']['goal_com']
        dist = position_euclidean_distance(self.unwrapped.sim.data.qpos[:3], goal_pos)
        reward = self.__prev_dist - dist
        self.__prev_dist = dist
        return reward
        # return -position_euclidean_distance(self.unwrapped.data.qpos[:3], self.goal_pos)
