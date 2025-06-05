from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading.pair_trading_env import PairTradingEnv
import pandas as pd


def train_ppo(prices: pd.DataFrame, pair: tuple[str, str], timesteps: int = 10000) -> PPO:
    env = DummyVecEnv([lambda: PairTradingEnv(prices, pair)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model

