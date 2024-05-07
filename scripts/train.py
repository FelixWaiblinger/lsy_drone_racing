from __future__ import annotations

import logging
# import time
from functools import partial
from pathlib import Path

# import fire
import numpy as np
import pybullet as p
import yaml
from munch import Munch, munchify
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync
from safe_control_gym.controllers.ppo.ppo import PPO
# from stable_baselines3.common.env_checker import check_env

from lsy_drone_racing.command import apply_sim_command
from lsy_drone_racing.utils import load_controller


logger = logging.getLogger(__name__)

config: str = "config/getting_started.yaml"
gui: bool = True
save_path = "./test"

# Load configuration and check if firmare should be used.
path = Path(config)
assert path.exists(), f"Configuration file not found: {path}"
with open(path, "r") as file:
    config = munchify(yaml.safe_load(file))
# Overwrite config options
config.quadrotor_config.gui = gui
CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
CTRL_DT = 1 / CTRL_FREQ

# Create environment.
assert config.use_firmware, "Firmware must be used for the competition."
FIRMWARE_FREQ = 500
pyb_freq = config.quadrotor_config["pyb_freq"]
assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
# The env.step is called at a firmware_freq rate, but this is not as intuitive to the end
# user, and so we abstract the difference. This allows ctrl_freq to be the rate at which the
# user sends ctrl signals, not the firmware.
config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ
env_func = partial(make, "quadrotor", **config.quadrotor_config)
wrapped_env = make("firmware", env_func, FIRMWARE_FREQ, CTRL_FREQ)
env = wrapped_env.env


# test environment
# check_env(env)

params = {
    # environment
    "rollout_batch_size": 1,
    "num_workers": 1,
    "deque_size": 1,
    # agent
    "hidden_dim": 256,
    "use_clipped_value": False,
    "clip_param": ...,
    "target_kl": ...,
    "entropy_coef": ...,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "opt_epochs": ...,
    "mini_batch_size": 1,
    # training
    "max_env_steps": 1000,
    "save_interval": 100,
    "num_checkpoints": 10,
    "eval_interval": 100,
    "eval_save_best": True,
    # running
    "rollout_steps": 10,
}

model = PPO(env_func, **params)

model.reset()

model.learn(env)

model.save(save_path)

model.close()
env.close()
