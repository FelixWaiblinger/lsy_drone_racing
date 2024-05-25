"""Training and Evaluation"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import fire
import yaml
from munch import munchify
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.wrapper import \
    DroneRacingWrapper, DroneRacingRewardWrapper, RewardWrapper


SAVE_PATH = "./test"
TASK = "train" # one of [train, retrain, eval]
TRAIN_STEPS = 100000
LOG_FOLDER = "./ppo_drones_tensorboard/"
LOG_NAME = "ppo_test"


def create_race_env(config_path: Path, gui: bool = False) -> DroneRacingWrapper:
    """Create the drone racing environment."""
    # Load configuration and check if firmare should be used.
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    with open(config_path, "r", encoding='utf-8') as file:
        config = munchify(yaml.safe_load(file))
    # Overwrite config options
    config.quadrotor_config.gui = gui
    CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
    # Create environment
    assert config.use_firmware, "Firmware must be used for the competition."
    pyb_freq = config.quadrotor_config["pyb_freq"]
    assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
    config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ

    def env_factory():
        env = partial(make, "quadrotor", **config.quadrotor_config)
        env = make("firmware", env, FIRMWARE_FREQ, CTRL_FREQ)
        env = DroneRacingWrapper(env, terminate_on_lap=True)
        env = DroneRacingRewardWrapper(env)
        # env = RewardWrapper(env)
        # check_env(env)
        return env

    # env = make_vec_env(
    #     env_factory,
    #     n_envs=8,
    #     vec_env_cls=DummyVecEnv, #SubprocVecEnv,
    #     # vec_env_kwargs={"start_method": "fork"}
    # )
    env = env_factory()

    return env


def train(
    config: str = "config/getting_started.yaml",
    gui: bool = False,
    resume: bool = False
):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / config
    env = create_race_env(config_path=config_path, gui=gui)
    # check_env(env)  # Sanity check to ensure the environment conforms to the sb3 API

    if resume:
        print("Continuing...")
        agent = PPO.load(SAVE_PATH, env)
    else:
        print("Training new agent...")
        agent = PPO("MlpPolicy", env, tensorboard_log=LOG_FOLDER)
    agent.learn(total_timesteps=TRAIN_STEPS, progress_bar=True, tb_log_name=LOG_NAME)
    agent.save(SAVE_PATH)


if __name__ == "__main__":
    if TASK == "train":
        fire.Fire(train)
    elif TASK == "retrain":
        fire.Fire(train, command=["--resume", "True"])
    else:
        path_to_config = Path(__file__).resolve().parents[1] / "config/getting_started.yaml"
        test_env = create_race_env(config_path=path_to_config, gui=True)
        model = PPO.load(SAVE_PATH, test_env)
        mean_reward, std_reward = evaluate_policy(
            model,
            model.get_env(),
            n_eval_episodes=10,
            deterministic=False
        )
        print(f"{mean_reward = }")
        print(f"{std_reward = }")
