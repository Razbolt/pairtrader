import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PairTradingEnv(gym.Env):
    """A simple pair trading environment for RL."""

    def __init__(self, prices: pd.DataFrame, pair: tuple[str, str]):
        super().__init__()
        self.prices = prices[list(pair)].dropna()
        self.pair = pair
        self.current_step = 0

        # Observation: price spread
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        # Actions: 0 -> hold, 1 -> long first/short second, 2 -> short first/long second
        self.action_space = spaces.Discrete(3)
        self.position = 0  # -1 short pair, 0 flat, 1 long pair
        self.entry_price = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        spread = self.prices.iloc[self.current_step, 0] - self.prices.iloc[self.current_step, 1]
        return np.array([spread], dtype=np.float32)

    def step(self, action: int):
        done = False
        reward = 0.0
        if action == 1 and self.position == 0:  # long first, short second
            self.position = 1
            self.entry_price = self._get_obs()[0]
        elif action == 2 and self.position == 0:  # short first, long second
            self.position = -1
            self.entry_price = self._get_obs()[0]
        elif action == 0 and self.position != 0:  # close position
            reward = self.position * (self.entry_price - self._get_obs()[0])
            self.position = 0

        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            done = True
        return self._get_obs(), reward, done, False, {}

