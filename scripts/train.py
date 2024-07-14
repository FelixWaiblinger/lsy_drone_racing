"""Training and Evaluation"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Type, Callable

import fire
import yaml
import torch
import numpy as np
from munch import munchify

from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO, SAC, DDPG,TD3,A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from lsy_drone_racing.constants import FIRMWARE_FREQ, CTRL_TIMESTEP
from lsy_drone_racing.newwrapper import DroneRacingWrapper, RewardWrapper

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv

logger = logging.getLogger(__name__)
LOG_FOLDER = "./ppo_drones_tensorboard/"
LOG_NAME = "ppo_gs_l3"#"ppo_large"
LOAD_PATH = "./ppo_gs_l3_l_5"#"./models/baseline_getting_started"
SAVE_PATH = "./ppo_gs_l3_l_5"
TRAJ_PATH = "./reference_trajectory_steps.yaml"
CONFI_PATH = "./config/level3.yaml"
TRAIN_STEPS = 3_000_000
N_ENVS = 4
END = "next9"

# NOTE: abbreviations
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
        drone_racing_env = RewardWrapper(drone_racing_env)
        return drone_racing_env
    # env = make_vec_env(
    #     lambda: env_factory(),
    #     n_envs = N_ENVS,
    #     vec_env_cls=SubprocVecEnv
    #     )
    
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


class ResetCallback(BaseCallback):
    def __init__(self, env, n_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        assert hasattr(env, "reset_done"), "Env has no 'reset_done' property!"
        self.env = env
        # RewardWrapper.DroneRacingWrapper.FirmwareWrapper.Quadrotor(BenchmarkEnv)
        self.bench_env = env.env.env.env
        self.seed = self.bench_env.RND_SEED
        self.n_episodes = n_episodes
        self.current_episode = 0

    def _on_step(self) -> bool:
        if self.env.reset_done:
            self.current_episode += 1

        if self.current_episode == self.n_episodes:
            print("reseeding...")
            self.bench_env.seed(self.seed)
            self.current_episode = 0

        return True


def train(gui: bool = False, resume: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / CONFI_PATH
    env = create_race_env(config_path=config_path, gui=gui)
    eval_env = create_race_env(config_path=config_path, gui=gui)

    save = "./best/" + END + "/"
    reset_callback = ResetCallback(env, n_episodes=10)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save,
        log_path=save + "logs/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    if resume:
        print("Continuing...")
        custom_objects = {
            'learning_rate': linear_schedule(0.00003, slope=0.5),#0.00001,
            # 'n_epochs': 15,
            # 'n_steps': 4096,
        }
        agent = PPO.load(LOAD_PATH, env, custom_objects=custom_objects)

    else:
        print("Training new agent...")
        #smaller lr or batch size, toy problem mit nur hovern (reward anpassen)
        #learing rate scheduler
        # onpolicy_kwargs = dict(
        #     activation_fn=torch.nn.Tanh,
        #     net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
        #     # features_extractor="mlp"
        #     # net_arch=[128, 128, dict(vf=[256], pi=[128, 256])]
        # )
        agent = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_FOLDER)#, policy_kwargs=onpolicy_kwargs)

    for _ in range(6):
        agent.learn(
            total_timesteps=TRAIN_STEPS // 6,
            progress_bar=True,
            tb_log_name=LOG_NAME,
            callback=[reset_callback, eval_callback]
        )
    agent.save(SAVE_PATH + "_" + str(END))
    print(f"Saved model {END}")


def evaluate():
    """Evaluate the trained model."""
    path_to_config = Path(__file__).resolve().parents[1] / CONFI_PATH
    test_env = create_race_env(config_path=path_to_config, gui=True)
    reset_callback = ResetCallback(test_env, n_episodes=5)
    model = PPO.load("ppo_gs_l3_m_1_lrl1", test_env)
    
    rewards, times = [], []
    for run in range(10):
        rewards.append(0)
        times.append(0)
        obs, _ = test_env.reset()
        for t in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            # reset_callback._on_step()
            obs, reward, terminated, truncated, _ = test_env.step(action)
            rewards[-1] += reward
            times[-1] = t * CTRL_TIMESTEP
            if terminated or truncated:
                break
        print(f"Run: {run} | R_total: {rewards[-1]:.2f} | Time: {times[-1]:.2f}")
    print(f"R_mean: {(sum(rewards) / 10):.2f}")

    # mean_reward, std_reward = evaluate_policy(
    #     model,
    #     model.get_env(),
    #     n_eval_episodes=10,
    #     deterministic=True,
    #     callback=reset_callback
    # )
    # print(f"{mean_reward = }")
    # print(f"{std_reward = }")


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
