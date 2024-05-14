""""""

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

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.wrapper import DroneRacingWrapper


logger = logging.getLogger(__name__)

SAVE_PATH = "./test2"
TRAIN = True
TRAIN_STEPS = 10000


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
    env_factory = partial(make, "quadrotor", **config.quadrotor_config)
    firmware_env = make("firmware", env_factory, FIRMWARE_FREQ, CTRL_FREQ)
    return DroneRacingWrapper(firmware_env, terminate_on_lap=True)


def train(config: str = "config/getting_started.yaml", gui: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / config
    env = create_race_env(config_path=config_path, gui=gui)
    check_env(env)  # Sanity check to ensure the environment conforms to the sb3 API
    agent = PPO("MlpPolicy", env, verbose=1)
    agent.learn(total_timesteps=TRAIN_STEPS)
    agent.save(SAVE_PATH)


if __name__ == "__main__":
    if TRAIN:
        fire.Fire(train)
    else:
        path_to_config = Path(__file__).resolve().parents[1] / "config/getting_started.yaml"
        test_env = create_race_env(config_path=path_to_config, gui=False)
        model = PPO.load(SAVE_PATH, test_env)
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=False)
        print(f"{mean_reward = }")
        print(f"{std_reward = }")
