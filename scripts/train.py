"""Training and Evaluation"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import fire
import yaml
import numpy as np
from munch import munchify
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO, DDPG, SAC
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

from lsy_drone_racing.utils import linear_schedule, draw_policy
from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.wrapper import DroneRacingWrapper, RewardWrapper #, HoverRewardWrapper


AGENT_PATH = "./gate_tracking_sac"
TRAIN_STEPS = 500_000
CONFIG = "config/level0.yaml"
LOG_FOLDER = "./ppo_drones_tensorboard/"
LOG_NAME = "gate_tracking_sac"


def create_race_env(config_path: Path, gui: bool = False) -> DroneRacingWrapper:
    """Create the drone racing environment."""

    def env_factory():
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
        env = partial(make, "quadrotor", **config.quadrotor_config)
        env = make("firmware", env, FIRMWARE_FREQ, CTRL_FREQ)

        return env

    # env = make_vec_env(
    #     lambda: MultiProcessingWrapper(RewardWrapper(DroneRacingWrapper(env_factory()))),
    #     n_envs=1,
    #     vec_env_cls=DummyVecEnv
    #     # vec_env_cls=SubprocVecEnv,
    #     # vec_env_kwargs={"start_method": "fork"}
    # )
    # env = VecCheckNan(env, raise_exception=True)
    env = RewardWrapper(DroneRacingWrapper(env_factory()))
    # env = HoverRewardWrapper(DroneRacingWrapper(env_factory()))

    return env


def start_training():
    """Create the environment create a new agent and train it"""
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / CONFIG
    env = create_race_env(config_path=config_path, gui=False)
    agent = SAC("MlpPolicy", env, tensorboard_log=LOG_FOLDER,
                # n_steps=64, batch_size=64,
                learning_rate=linear_schedule(0.001, 0.5))

    print("Training new agent...")
    try:
        agent.learn(
            total_timesteps=TRAIN_STEPS,
            progress_bar=True,
            tb_log_name=LOG_NAME
        )
    except Exception as e:
        print(e)
    agent.save(AGENT_PATH)


def continue_training():
    """Create the environment, load a pretrained agent and continue training"""
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / CONFIG
    env = create_race_env(config_path=config_path, gui=False)
    agent = PPO.load(AGENT_PATH, env)

    print("Continuing...")
    agent.learn(total_timesteps=TRAIN_STEPS, progress_bar=True, tb_log_name=LOG_NAME)
    agent.save(AGENT_PATH)


def evaluate():
    """Create the environment, load a pretrained agent and evaluate it"""
    logging.basicConfig(level=logging.INFO)
    path_to_config = Path(__file__).resolve().parents[1] / CONFIG
    env = create_race_env(config_path=path_to_config, gui=True)
    model = SAC.load(AGENT_PATH, env)

    obs, _ = env.reset()
    # draw_policy(model, obs, size=(3, 3, 2))
    reward, episodes, state = 0, 1, 0
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        # action, state = hardcoded_predict(state, obs)
        # action[:3] /= 0.1 # ensure xyz is in [-1, 1]
        obs, rew, ter, tru, inf = env.step(action)
        # print(f"{obs[:3]}")
        reward += rew
        if ter or tru:
            episodes += 1
            state = 0
            obs, _ = env.reset()

    print(f"rew/episode: {reward / episodes}")


def hardcoded_predict(state, obs):
    """Test"""
    first_gate = np.array([0.45, -1.0, 0, 0], dtype=np.float32)
    some_pos = np.array([0, -2, 0.7, 0], dtype=np.float32)
    second_gate = np.array([1.0, -1.55, 1, 0], dtype=np.float32)

    if state == 0:
        action = first_gate - obs[[0, 1, 2, 5]]
        dist = np.linalg.norm(first_gate[:2] - obs[:2], ord=2)
        if dist < 0.15:
            print("state 0 passed")
            state += 1
    elif state == 1:
        action = some_pos - obs[[0, 1, 2, 5]]
        dist = np.linalg.norm(some_pos[:2] - obs[:2], ord=2)
        if dist < 0.15:
            print("state 1 passed")
            state += 1
    elif state == 2:
        action = second_gate - obs[[0, 1, 2, 5]]
        dist = np.linalg.norm(first_gate[:2] - obs[:2], ord=2)
        if dist < 0.15:
            print("Woop woop")

    action /= np.max(action) # normalize
    return action, state


if __name__ == "__main__":

    # fire.Fire(start_training)

    # fire.Fire(continue_training)

    fire.Fire(evaluate)
