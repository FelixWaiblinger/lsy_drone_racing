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
from stable_baselines3 import PPO

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
        self.action_scale = np.array([1, 1, 1, np.pi])
        self.state = 0

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        self.model = PPO.load("models/ppo_gs_l3_l_5")

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
        #########################
        # REPLACE THIS (START) ##
        #########################

        # avoid illegal gate poses
        clip = lambda x, y: np.clip(x, -y, y)
        gate_limit = np.array([5, 5, 5, np.pi])
        obst_limit = np.array([5, 5, 5])

        gates = obs[12:28].reshape((4,4))
        obsts = obs[32:44].reshape((4,3))
        gate_poses = np.array([clip(gate, gate_limit) for gate in gates])
        obst_poses = np.array([clip(obst, obst_limit) for obst in obsts])
        obs[12:28] = gate_poses.flatten().astype(np.float32)
        obs[32:44] = obst_poses.flatten().astype(np.float32)
        
        # state machine
        zero = np.zeros(3)
        self.state = 2 #self._check_state(ep_time, info)
        # init -> takeoff
        if self.state == 0:
            command_type = Command.TAKEOFF
            args = [0.1, 1]
        # take off -> wait
        elif self.state == 1:
            command_type = Command.NONE
            args = []
        # wait -> fly
        elif self.state == 2:
            action, _ = self.model.predict(obs, deterministic=True)
            action[3] = 0
            action = self._action_transform(action).astype(float)
            command_type = Command.FULLSTATE
            args = [action[:3], zero, zero, action[3], zero, ep_time]
        # fly -> notify
        elif self.state == 3:
            command_type = Command.NOTIFYSETPOINTSTOP
            args = [] # TODO replace by correct
        elif self.state == 4:
            command_type = Command.LAND
            args = [0, 3]
        else:
            command_type = Command.NONE
            args = []


        #########################
        # REPLACE THIS (END) ####
        #########################

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
        self._drone_pose = obs[[0, 1, 2, 5]]

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        """
        pass

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
    
    def _check_state(self, time, info):
        # print(self.state)
        if self.state == 0: # initialization state
            return 1
        elif self.state == 1 and time < 1: # take off state
            return 2
        elif self.state == 2 and time < 5:#info["task_completed"]: # flying state
            return 3
        elif self.state == 3: # notify state
            return 4
        elif self.state == 4: # landing state
            return 5
        else: # finished state
            return self.state