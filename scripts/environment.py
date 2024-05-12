"""Environment wrapper for gymnasium compatibility"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lsy_drone_racing.command import apply_sim_command


class Environment(gym.Env):
    """Gymnasium compatible environment wrapper"""

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 10}

    def __init__(self, env, agent, render_mode: str | None=None) -> None:
        """Create a gymnasium environment wrapper"""
        self.env = env
        self.agent = agent
        self.render_mode = render_mode
        self.ep_time = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(12,)),
            "gates": ...,
            "obstacles": ...
        })
        super().__init__()

    def reset(self, *, seed=None, options=None):
        """Reset the environment (between episodes)"""
        super().reset(seed=seed, options=options)
        self.agent.reset()
        self.ep_time = 0

        return np.zeros(12).astype(np.float32), {}

    def step(self, action):
        """Perform the chosen action and update the environment one step"""
        delta_time, command_type, args = action

        # agent performs action
        apply_sim_command(self.env, command_type, args)

        # environment is updated
        obs, reward, done, info, action = self.env.step(self.ep_time, action)
        self.ep_time += delta_time

        # observations, reward, terminated, truncated, info
        return obs, reward, done, False, info

    def render(self):
        """Render the next frame"""

    def close(self):
        """Clean-up and close this environment"""
        self.env.close()

    def _get_observations(self):
        pass

    def _get_info(self):
        pass
