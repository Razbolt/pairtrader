from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from trading.pair_trading_env import PairTradingEnv
import pandas as pd


def train_dqn(prices: pd.DataFrame, pair: tuple[str, str], timesteps: int = 10000) -> DQN:
    env = DummyVecEnv([lambda: PairTradingEnv(prices, pair)])
    model = DQN("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model

