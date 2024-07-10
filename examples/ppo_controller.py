from __future__ import annotations  # Python 3.10 type hints
import os
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.command import Command
from lsy_drone_racing.rotations import map2pi
from lsy_drone_racing.controller import BaseController

class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
        """
        super().__init__(initial_obs, initial_info, buffer_size, verbose)
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size
        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
        self._drone_pose = None
        self.action_scale = np.array([1.0,1.0,1.0, np.pi])
        self.state = None
        self._takeoff = False
        self.initial_info = initial_info        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        # NOTE: no need to pass the environment to PPO.load
        # get the the relative path of the model
        MODEL = "hover"
        # global PATH directory
        PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", MODEL))
        self.model = PPO.load(PATH)

    def reset(self):
        self._drone_pose = self.initial_obs[[0, 1, 2, 5]]
        
        
    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        zero = np.zeros(3)
        # Default command
        command_type = Command.NONE
        args = []
        if not self._takeoff:
            command_type = Command.TAKEOFF
            args = [0.06, 1]  # Height, duration
            self._takeoff = True  # Only send takeoff command once
        else:
            if ep_time - 2 > 0 and not done:
                # Get action from the model
                action, _ = self.model.predict(obs, deterministic=True)
                action[3] = 0
                action = self._action_transform(action).astype(float)
                command_type = Command.FULLSTATE
                args = [action[:3], zero, zero, action[3], zero, ep_time]
            elif done:
                if not self._setpoint_land:
                    command_type = Command.NOTIFYSETPOINTSTOP
                    args = []
                    self._setpoint_land = True
                elif not self._land:
                    command_type = Command.GOTO
                    args = [self._drone_pose[:3], 0.0, 2, True]  # pos, yaw, duration, relative
                    self._land = True  # Send landing command only once
                elif self._land:
                    command_type = Command.FINISHED
                    args = []
                else:
                    command_type = Command.NONE
                    args = []

        return command_type, args

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        Args:
            action: Most recent applied action.
            obs: Most recent observation of the quadrotor state.
            reward: Most recent reward.
            done: Most recent done flag.
            info: Most recent information dictionary.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        self._drone_pose = obs[[0, 1, 2, 5]]
        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        # Implement some learning algorithm here if needed

        #########################
        # REPLACE THIS (END) ####
        #########################

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        #########################
        # REPLACE THIS (END) ####
        #########################

    def _action_transform(self, action: np.ndarray) -> np.ndarray:
        """Transform the action to the format expected by the firmware env.

        Args:
            action: The action to transform.

        Returns:
            The transformed action.
        """
        action = self._drone_pose + (action * self.action_scale)
        action[3] = map2pi(action[3])  # Ensure yaw is in [-pi, pi]
        return action

    def _action_transform_abs(self, action: np.ndarray) -> np.ndarray:
        """Transform the action to the format expected by the firmware env.

        Args:
            action: The action to transform.

        Returns:
            The transformed action.
        """
        action = action * self.action_scale
        action[3] = map2pi(action[3])
        return action
