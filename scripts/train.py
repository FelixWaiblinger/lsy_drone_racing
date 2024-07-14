"""Training and Evaluation"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
import numpy as np

import fire
import yaml
from munch import munchify
from typing import TYPE_CHECKING, Type, Callable

from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO, SAC, DDPG,TD3,A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies  import  ActorCriticPolicy as a2cppoMlpPolicy
from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.wrapper import DroneRacingWrapper, GateWrapper
#from lsy_drone_racing.racewrapper import RaceWrapper, DroneRacingWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import torch

logger = logging.getLogger(__name__)
LOG_FOLDER = "./ppo_drones_tensorboard/"
LOG_NAME = "level1_plot"
SAVE_PATH = "./baseline_level1"
CONFI_PATH = "./config/level1.yaml"
TRAIN_STEPS = 1000_000
N_ENVS = 5

def create_race_env(config_path: Path, gui: bool = False) :

    def env_factory():
    #    """Create the drone racing environment."""
        
    # Load configuration and check if firmare should be used.
        assert config_path.exists(), f"Configuration file not found: {config_path}"
        with open(config_path, "r", encoding='utf-8') as file:
            config = munchify(yaml.safe_load(file))
        # Overwrite config optionsl
        config.quadrotor_config.gui = gui
        CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
        # Create environment
        assert config.use_firmware, "Firmware must be used for the competition."
        pyb_freq = config.quadrotor_config["pyb_freq"]
        print(config.quadrotor_config.done_on_completion)
        assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
        config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ
        env_factory = partial(make, "quadrotor", **config.quadrotor_config)
        firmware_env = make("firmware", env_factory, FIRMWARE_FREQ, CTRL_FREQ)
        drone_racing_env = DroneRacingWrapper(firmware_env, terminate_on_lap=True)
        drone_racing_env = GateWrapper(drone_racing_env)
        return drone_racing_env
    env = make_vec_env(
        lambda: env_factory(),
        n_envs = N_ENVS,
        vec_env_cls=SubprocVecEnv
        )

    return env



def linear_schedule(initial_value: float, slope: float=1) -> Callable[[float], float]:
    """Linear learning rate schedule (current learning rate depending on
    remaining progress)
    """

    assert 0 < slope < 1, f"Invalid slope {slope} in schedule!"

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0"""

        return initial_value * (1 - slope + slope * progress_remaining )

    return func


def train(
    gui: bool = False,
    resume: bool = False
):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / CONFI_PATH
    env = create_race_env(config_path=config_path, gui=gui)

    if resume:
        print("Continuing...")
        agent = PPO.load(SAVE_PATH, env)
    else:
        print("Training new agent...")
        #smaller lr or batch size, toy problem mit nur hovern (reward anpassen)
        #learing rate scheduler
        policy_kwargs = dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],
            activation_fn=torch.nn.ReLU,
        )
        agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_FOLDER, policy_kwargs= policy_kwargs)
    agent.learn(total_timesteps=TRAIN_STEPS, progress_bar=True,tb_log_name=LOG_NAME)
    agent.save(SAVE_PATH)


def evaluate():
    """Evaluate the trained model."""
    path_to_config = Path(__file__).resolve().parents[1] / CONFI_PATH
    test_env = create_race_env(config_path=path_to_config, gui=True)
    model = PPO.load(SAVE_PATH, test_env)

    # NOTE: from Martin Schuck<s
    
    '''
    for i in range(10):
        total_reward = 0
        obs, info = test_env.reset()
        done = False
        rew = []
        for j in range(1000):
            action = np.array([1,-0.5,1,0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = test_env.step(action)
            rew.append(reward)
            done = terminated or truncated
            if done:
                break
        print(sum(rew) / len(rew))

    '''

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True)
    print(f"{mean_reward = }")
    print(f"{std_reward = }")
    
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