"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory


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

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()
        # PID Controller gains
        self.kp = 0.001
        self.kd = 0.000
        self.ki = 0.000
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_time = 0.0
        self._take_off = False
        self._setpoint_land = False
        self._land = False

    def pid_controller(self, target_pos, current_pos):
        """PID controller for position control."""
        error = target_pos - current_pos
        self.integral += error * self.CTRL_TIMESTEP
        derivative = (error - self.prev_error) / self.CTRL_TIMESTEP

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        return output

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

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
        iteration = int(ep_time * self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handcrafted solution for getting_stated scenario.

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.3, 2]  # Height, duration
            self._take_off = True  # Only send takeoff command once
        else:
            step = iteration - 2 * self.CTRL_FREQ  # Account for 2s delay due to takeoff
            if ep_time - 2 > 0 and step < len(self.NOMINAL_GATES):
                #get the target position x, y, z
                target_pos = [0.45, -1.0, 0.5]
                #target_pos = np.array([target_pos[0] - obs[0], target_pos[1] - obs[2], target_pos[2]-obs[4]])
                print(f"Target position: {target_pos}")
                current_pos = obs[0], obs[2], obs[4]  # x, y, z
                print(f"Current position: {current_pos}")

                #save current position into a list
                current_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])
               
                pid_output = self.pid_controller(target_pos, current_pos)
                print(pid_output)
                # Apply PID output to get the next desired position
                next_desired_pos = [target_pos[i] + pid_output[i] for i in range(3)]
                print(f"Next desired position: {next_desired_pos}")
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.0
                target_rpy_rates = np.zeros(3)
                command_type = Command.FULLSTATE
                args = [next_desired_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            else:
                command_type = Command.NONE
                args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args
