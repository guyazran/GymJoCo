from typing import Callable, Union, Dict

import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.core import ActType
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv

TIMESTEP_OBS_KEY = '__SCRIPTED_POLICY_frame'
ORIG_OBS_KEY = 'observation'


class ScriptedAgent(BaseAlgorithm):
    def __init__(self, script: Callable[[int], ActType], env: Union[GymEnv, str, None]):
        super().__init__(policy=ScriptedPolicy,
                         policy_kwargs=dict(script=script),
                         env=env,
                         learning_rate=0)
        self._setup_model()
        self.frame = 0

    def _setup_model(self):
        if isinstance(self.observation_space, spaces.Dict):
            s = self.observation_space.spaces.copy()  # avoid mutating original dict
            s[TIMESTEP_OBS_KEY] = spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32)
            self.observation_space = spaces.Dict(s)
        else:
            self.observation_space = spaces.Dict({
                TIMESTEP_OBS_KEY: spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                ORIG_OBS_KEY: self.observation_space
            })

        self.policy = self.policy_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            **self.policy_kwargs
        )

    def learn(self, *args, **kwargs):
        return self

    def predict(self, observation: Union[np.ndarray, Dict[str, np.ndarray]], *args, **kwargs):
        if isinstance(observation, dict):
            observation = observation.copy()  # avoid mutating original dict
            observation[TIMESTEP_OBS_KEY] = self.frame
        else:
            observation = {
                TIMESTEP_OBS_KEY: self.frame,
                ORIG_OBS_KEY: observation
            }

        return self.policy.predict(observation, *args, **kwargs)

    def reset_frames(self):
        self.frame = 0

    def step_callback(self, locals_, globals_):
        if locals_['dones'][0]:
            self.reset_frames()
        else:
            self.frame += 1

    def run(self, n_episodes=1, callback=None):

        # allow additional callback
        if callback is None:  # use step counting callback
            callback = self.step_callback
        else:  # call callback after step count
            def _callback(*args, **kwargs):
                self.step_callback(*args, **kwargs)
                callback(*args, **kwargs)

        return evaluate_policy(self, self.env,
                               n_eval_episodes=n_episodes,
                               render=self.env.render_mode == 'human',
                               callback=self.step_callback,
                               return_episode_rewards=True,
                               warn=True)


class ScriptedPolicy(BasePolicy):
    def __init__(self, script, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.script = script

    def _predict(self, obs, *args, **kwargs) -> th.Tensor:
        return th.as_tensor(self.script(obs[TIMESTEP_OBS_KEY]))
