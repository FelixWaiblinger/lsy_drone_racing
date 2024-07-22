"""Training and Evaluation"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Callable

import fire
import yaml
from munch import munchify
from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv

from safe_control_gym.utils.registration import make
from lsy_drone_racing.constants import FIRMWARE_FREQ, CTRL_TIMESTEP
from lsy_drone_racing.wrapper import \
    DroneRacingWrapper, RewardWrapper, MultiProcessingWrapper


logger = logging.getLogger(__name__)
LOG_FOLDER = "./ppo_drones_tensorboard/"
LOG_NAME = "ppo_gs_l3"
LOAD_PATH = "./ppo_gs_l3_l"
SAVE_PATH = "./baseline_level3"
CONFIG_PATH = "./config/level3.yaml"
TRAIN_STEPS = 3_000_000
N_ENVS = 4

# NOTE: abbreviations for hyperparameter optimization / ablation studies
# gs            = getting started
# l3            = level3
# GOOD  s/m/l   = num iterations on same agent
# GOOD  br      = just no bodyrate punishment (included in below configs aswell)
# MORE  as      = action_scale to 0.8
# BAD   ep      = n_epochs to 5
# BAD   st      = n_steps to 1024
# BAD   ga      = gamma to 0.95
# BAD   lu      = learning_rate to 0.0001 (instead of 0.00003)
# GOOD  ld      = learning_rate to 0.00001 (instead of 0.00003)
# BAD   en      = entropy coefficient to 0.05
# GOOD  ns      = n_steps to 4096
# BAD   rp      = train 10 times for 200k each

def create_race_env(config_path: Path, gui: bool = False):
    """Create a wrapped environment for ready for training."""

    def env_factory():
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
        drone_racing_env = DroneRacingWrapper(firmware_env, terminate_on_lap=True)
        drone_racing_env = RewardWrapper(drone_racing_env)
        drone_racing_env = MultiProcessingWrapper(drone_racing_env)
        return drone_racing_env
    
    # NOTE: uncomment if multiprocessing should be used
    # env = make_vec_env(
    #     lambda: env_factory(),
    #     n_envs = N_ENVS,
    #     vec_env_cls=SubprocVecEnv
    # )
    
    env = env_factory()

    return env


def linear_schedule(initial_value: float, slope: float=1) -> Callable[[float], float]:
    """Linear learning rate schedule (current learning rate depending on
    remaining progress)
    """

    assert 0 <= slope <= 1, f"Invalid slope {slope} in schedule!"

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0"""

        return initial_value * (1 - slope + slope * progress_remaining )

    return func


def train(gui: bool = False, resume: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / CONFIG_PATH
    env = create_race_env(config_path=config_path, gui=gui)

    # NOTE: used for saving models inbetween training sessions
    # eval_env = create_race_env(config_path=config_path, gui=gui)
    # save = "./best/" + END + "/"
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=save,
    #     log_path=save + "logs/",
    #     eval_freq=1000,
    #     deterministic=True,
    #     render=False
    # )

    # NOTE: train a baseline model for additional TRAIN_STEPS
    if resume:
        print("Continuing...")
        custom_objects = {
            'learning_rate': linear_schedule(0.00003, slope=0.5),
        }
        agent = PPO.load(LOAD_PATH, env, custom_objects=custom_objects)

    # NOTE: train a new model from scratch
    else:
        print("Training new agent...")
        agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_FOLDER)

    agent.learn(
        total_timesteps=TRAIN_STEPS,
        progress_bar=True,
        tb_log_name=LOG_NAME,
    )

    agent.save(SAVE_PATH)
    print(f"Saved model {SAVE_PATH}")


def evaluate():
    """Evaluate the trained model."""
    path_to_config = Path(__file__).resolve().parents[1] / CONFIG_PATH
    test_env = create_race_env(config_path=path_to_config, gui=True)
    model = PPO.load(SAVE_PATH, test_env)

    # NOTE: evaluate a model over 10 episodes of maximum 1000 steps each
    rewards, times = [], []
    for run in range(10):
        rewards.append(0)
        times.append(0)
        obs, _ = test_env.reset()
        for t in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            rewards[-1] += reward
            times[-1] = t * CTRL_TIMESTEP
            if terminated or truncated:
                break
        print(f"Run: {run} | R_total: {rewards[-1]:.2f} | Time: {times[-1]:.2f}")
    print(f"R_mean: {(sum(rewards) / 10):.2f}")


def main(task: str):
    """Main function to handle task selection."""
    if task == "train":
        train()
    elif task == "retrain":
        train(resume=True)
    elif task == "eval":
        evaluate()
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    fire.Fire(main)
