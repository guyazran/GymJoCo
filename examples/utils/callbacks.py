from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean

from gymjoco.tasks.rearrangement.scoring import discounted_return


class LogMeanEpisodeValueCallback(BaseCallback, ABC):
    @abstractmethod
    def _get_value(self, env_i):
        pass

    def __init__(self, log_name: str, episode_agg=np.mean, verbose: int = 0):
        super().__init__(verbose)
        self.episode_agg = episode_agg

        # key of log item
        # e.g. the info "inner_reward" aggregated with sum will be:
        #     rollout/mean_ep_sum_inner_reward
        self.__log_key = f"rollout/mean_ep_{self.episode_agg.__name__}_{log_name}"

    def _init_callback(self) -> None:
        self.__ep_infos = [[] for _ in range(self.model.n_envs)]
        self.__agg_ep_infos = deque(maxlen=100)
        self.__is_off_policy = isinstance(self.model, OffPolicyAlgorithm)

        self.__rollout_count = 0

    def _on_training_start(self) -> None:
        self.log_interval = self.locals['log_interval']

    def _on_step(self) -> bool:
        for i in range(self.model.n_envs):
            vi = self._get_value(env_i=i)

            if vi is not None:
                self.__ep_infos[i].append(vi)

            if self.locals['dones'][i]:
                self.__agg_ep_infos.append(self.episode_agg(self.__ep_infos[i]))
                self.__ep_infos[i] = []

                # dump rule for off policy
                if self.__is_off_policy:
                    self.__maybe_dump(iterations=len(self.__agg_ep_infos))  # num iters == num episodes

        return True

    def _on_rollout_end(self):
        self.__rollout_count += 1

        # dump rule for on policy
        if not self.__is_off_policy:
            self.__maybe_dump(self.__rollout_count)  # num iters == num rollouts

    def __maybe_dump(self, iterations):
        if (self.log_interval is not None and  # must have a log interval
                len(self.__agg_ep_infos) > 0 and  # must complete at least one episode
                iterations % self.log_interval == 0):  # num iters and log interval coincide
            self.logger.record(self.__log_key, safe_mean(self.__agg_ep_infos))


class LogInfoCallback(LogMeanEpisodeValueCallback):
    def __init__(self, info_key, episode_agg=np.mean, verbose: int = 0):
        super().__init__(info_key, episode_agg, verbose)
        self.info_key = info_key

    def _get_value(self, env_i) -> bool:
        info = self.locals['infos'][env_i]
        return info[self.info_key] if self.info_key in info else 0


class LogAgentRewardCallback(LogMeanEpisodeValueCallback):
    def __init__(self, compute_discounted_return=False, verbose: int = 0):
        if compute_discounted_return:
            episode_agg = lambda x: discounted_return(x, self.model.gamma)
        else:
            episode_agg = np.sum
        super().__init__('agent_reward', episode_agg, verbose)

    def _get_value(self, env_i):
        return self.locals['rewards'][env_i]
