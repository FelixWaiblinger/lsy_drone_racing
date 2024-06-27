"""Wrapper to make the environment compatible with the gymnasium API.

The drone simulator does not conform to the gymnasium API, which is used by most RL frameworks. This
wrapper can be used as a translation layer between these modules and the simulation.

RL environments are expected to have a uniform action interface. However, the Crazyflie commands are
highly heterogeneous. Users have to make a discrete action choice, each of which comes with varying
additional arguments. Such an interface is impractical for most standard RL algorithms. Therefore,
we restrict the action space to only include FullStateCommands.

We also include the gate pose and range in the observation space. This information is usually
available in the info dict, but since it is vital information for the agent, we include it directly
in the observation space.

Warning:
    The RL wrapper uses a reduced action space and a transformed observation space!
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.error import InvalidAction
from gymnasium.spaces import Box
from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper

from lsy_drone_racing.rotations import map2pi
import yaml

logger = logging.getLogger(__name__)

class DroneRacingWrapper(Wrapper):
    """Drone racing firmware wrapper to make the environment compatible with the gymnasium API.

    In contrast to the underlying environment, this wrapper only accepts FullState commands as
    actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: FirmwareWrapper, terminate_on_lap: bool = True):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            terminate_on_lap: Stop the simulation early when the drone has passed the last gate.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        super().__init__(env)
        # Patch the FirmwareWrapper to add any missing attributes required by the gymnasium API.
        self.env = env
        # Unwrapped attribute is required for the gymnasium API. Some packages like stable-baselines
        # use it to check if the environment is unique. Therefore, we cannot use None, as None is
        # None returns True and falsely indicates that the environment is not unique. Lists have
        # unique id()s, so we use lists as a dummy instead.
        self.env.unwrapped = []
        self.env.render_mode = None

        # Gymnasium env required attributes
        # Action space:
        # [x, y, z, yaw]
        # x, y, z)  The desired position of the drone in the world frame.
        # yaw)      The desired yaw angle.
        # All values are scaled to [-1, 1]. Transformed back, x, y, z values of 1 correspond to 5m.
        # The yaw value of 1 corresponds to pi radians.
        self.action_scale = np.array([1, 1, 1, np.pi])
        self.action_space = Box(-1, 1, shape=(4,), dtype=np.float32)

        # Observation space:
        # [drone_xyz, drone_rpy, drone_vxyz, drone vrpy, gates_xyz_yaw, gates_in_range,
        # obstacles_xyz, obstacles_in_range, gate_id]
        # drone_xyz)  Drone position in meters.
        # drone_rpy)  Drone orientation in radians.
        # drone_vxyz)  Drone velocity in m/s.
        # drone_vrpy)  Drone angular velocity in rad/s.
        # gates_xyz_yaw)  The pose of the gates. Positions are in meters and yaw in radians. The
        #       length is dependent on the number of gates. Ordering is [x0, y0, z0, yaw0, x1,...].
        # gates_in_range)  A boolean array indicating if the drone is within the gates' range. The
        #       length is dependent on the number of gates.
        # obstacles_xyz)  The pose of the obstacles. Positions are in meters. The length is
        #       dependent on the number of obstacles. Ordering is [x0, y0, z0, x1,...].
        # obstacles_in_range)  A boolean array indicating if the drone is within the obstacles'
        #       range. The length is dependent on the number of obstacles.
        # gate_id)  The ID of the current target gate. -1 if the task is completed.
        n_gates = env.env.NUM_GATES
        n_obstacles = env.env.n_obstacles
        # Velocity limits are set to 10 m/s for the drone and 10 rad/s for the angular velocity.
        # While drones could go faster in theory, it's not safe in practice and we don't allow it in
        # sim either.
        # NOTE: target_limits = [5,5,5] for the target position
        #euclidean_distance_limit = np.linalg.norm([5,5,5], ord=2)
        #target_limits = [5,5,5]
        drone_limits = [5, 5, 5, np.pi, np.pi, np.pi, 10, 10, 10, 10, 10, 10]
        gate_limits = [5, 5, 5, np.pi] * n_gates + [1] * n_gates  # Gate poses and range mask
        obstacle_limits = [5, 5, 5] * n_obstacles + [1] * n_obstacles  # Obstacle pos and range mask
        obs_limits = drone_limits + gate_limits + obstacle_limits + [n_gates] #+ [euclidean_distance_limit]   # [1] for gate_id
        obs_limits_high = np.array(obs_limits)
        obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

        self.pyb_client_id: int = env.env.PYB_CLIENT
        # Config and helper flags
        self.terminate_on_lap = terminate_on_lap
        self._reset_required = False
        # The original firmware wrapper requires a sim time as input to the step function. This
        # breaks the gymnasium interface. Instead, we keep track of the sim time here. On each step,
        # it is incremented by the control time step. On env reset, it is reset to 0.
        self._sim_time = 0.0
        self._drone_pose = None
        # The firmware quadrotor env requires the rotor forces as input to the step function. These
        # are zero initially and updated by the step function. We automatically insert them to
        # ensure compatibility with the gymnasium interface.
        # TODO: It is not clear if the rotor forces are even used in the firmware env. Initial tests
        #       suggest otherwise.
        self._f_rotors = np.zeros(4)

    @property
    def time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._sim_time

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            seed: The random seed to use for the environment. Not used in this wrapper.
            options: Additional options to pass to the environment. Not used in this wrapper.

        Returns:
            The initial observation and info dict of the next episode.
        """
    
        self._reset_required = False
        self._sim_time = 0.0
        self._f_rotors[:] = 0.0
        obs, info = self.env.reset()
        # Store obstacle height for observation expansion during env steps.
        obs = self.observation_transform(obs, info).astype(np.float32)
        self._drone_pose = obs[[0, 1, 2, 5]]
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        assert not self._reset_required, "Environment must be reset before taking a step"
        if action not in self.action_space:
            # Wrapper has a reduced action space compared to the firmware env to make it compatible
            # with the gymnasium interface and popular RL libraries.
            raise InvalidAction(f"Invalid action: {action}")
        action = self._action_transform(action)
        assert action.shape[-1] == 4, "Action must have shape (..., 4)"
        # The firmware does not use the action input in the step function
        zeros = np.zeros(3)
        self.env.sendFullStateCmd(action[:3], zeros, zeros, action[3], zeros, self._sim_time)
        # The firmware quadrotor env requires the sim time as input to the step function. It also
        # returns the desired rotor forces. Both modifications are not part of the gymnasium
        # interface. We automatically insert the sim time and reuse the last rotor forces.
        obs, reward, done, info, f_rotors = self.env.step(self._sim_time, action=self._f_rotors)
        self._f_rotors[:] = f_rotors
        # We set truncated to True if the task is completed but the drone has not yet passed the
        # final gate. We set terminated to True if the task is completed and the drone has passed
        # the final gate.
        terminated, truncated = False, False
        if info["task_completed"] and info["current_gate_id"] != -1:
            truncated = True
        elif self.terminate_on_lap and info["current_gate_id"] == -1:
            info["task_completed"] = True
            terminated = True
        elif self.terminate_on_lap and done:  # Done, but last gate not passed -> terminate
            terminated = True
        # Increment the sim time after the step if we are not yet done.
        if not terminated and not truncated:
            self._sim_time += self.env.ctrl_dt
        # NOTE: add time to info dict
        info["time"] = self._sim_time
        obs = self.observation_transform(obs, info).astype(np.float32)
        self._drone_pose = obs[[0, 1, 2, 5]]
        if obs not in self.observation_space:
            terminated = True
        self._reset_required = terminated or truncated
        return obs, reward, terminated, truncated, info

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
    
    def render(self):
        """Render the environment.

        Used for compatibility with the gymnasium API. Checks if PyBullet was launched with an
        active GUI.

        Raises:
            AssertionError: If PyBullet was not launched with an active GUI.
        """
        assert self.pyb_client_id != -1, "PyBullet not initialized with active GUI"

    @staticmethod
    def observation_transform(obs: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Transform the observation to include additional information.

        Args:
            obs: The observation to transform.
            info: Additional information to include in the observation.

        Returns:
            The transformed observation.
        """
        drone_pos = obs[0:6:2]
        drone_vel = obs[1:6:2]
        drone_rpy = obs[6:9]
        drone_ang_vel = obs[8:11]
        #get the time of the simulation
       
        obs = np.concatenate(
            [
                drone_pos,
                drone_rpy,
                drone_vel,
                drone_ang_vel,
                info["gates_pose"][:, [0, 1, 2, 5]].flatten(),
                info["gates_in_range"],
                info["obstacles_pose"][:, :3].flatten(),
                info["obstacles_in_range"],
                [info["current_gate_id"]]
                #[0]
            ]
        )
        return obs

class DroneRacingObservationWrapper:
    """Wrapper to transform the observation space the firmware wrapper.

    This wrapper matches the observation space of the DroneRacingWrapper. See its definition for
    more details. While we want to transform the observation space, we do not want to change the API
    of the firmware wrapper. Therefore, we create a separate wrapper for the observation space.

    Note:
        This wrapper is not a subclass of the gymnasium ObservationWrapper because the firmware is
        not compatible with the gymnasium API.
    """

    def __init__(self, env: FirmwareWrapper):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        self.env = env
        self.pyb_client_id: int = env.env.PYB_CLIENT

    def __getattribute__(self, name: str) -> Any:
        """Get an attribute from the object.

        If the attribute is not found in the wrapper, it is fetched from the firmware wrapper.

        Args:
            name: The name of the attribute.

        Returns:
            The attribute value.
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.np.array([x, y, z]), ord=2)

        Returns:
            The transformed observation and the info dict.
        """
        obs, info = self.env.reset(*args, **kwargs)
        obs = DroneRacingWrapper.observation_transform(obs, info)
        return obs, info

    def step(
        self, *args: Any, **kwargs: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, dict, np.ndarray]:
        """Take a step in the current environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The transformed observation and the info dict.
        """
        obs, reward, done, info, action = self.env.step(*args, **kwargs)
        obs = DroneRacingWrapper.observation_transform(obs, info)
        return obs, reward, done, info, action

class HoverRewardWrapper(Wrapper):
    """Wrapper to alter the default reward function from the environment for RL training."""

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)


    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The initial observation of the next episode.
        """
        obs, info = self.env.reset(*args, **kwargs)
        # Delete CasADi models to enable multiprocessed environments. TODO: Put this in a separate
        # wrapper.
        del info["symbolic_model"]
        del info["symbolic_constraints"]
        return obs, info


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict
    ) -> float:
        """Compute the reward for the current step.

        Args:
            obs: The current observation.
            reward: The reward from the environment.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: Additional information from the environment.

        Returns:
            The computed reward.

        """
        #print info keys
        distance = np.linalg.norm(obs[:3] - np.ones(3), ord=2)
        distance_penalty = np.exp(-distance)
        collision = terminated 
        crash_penalty = -1 if collision else 0
        print(info["time"])
        return distance_penalty  + crash_penalty


class FollowTrajRewardWrapper(Wrapper):
    """Wrapper to alter the default reward function from the environment for RL training."""

    def __init__(self, env: Env, yaml_path: str):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            yaml_path: Path to the YAML file containing the trajectory.
        """
        self.yaml_path = yaml_path
        self.trajectory_points = self.load_trajectory()
        super().__init__(env)
        self.current_step = 0  # Initialize the step counter
        self.previous_position = None
        self.start_waypoint = self.trajectory_points[self.current_step][1:]

    def load_trajectory(self):
        """Load the trajectory from a YAML file and return the trajectory points."""
        with open(self.yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        steps = data['steps']
        x_values = data['ref_x']
        y_values = data['ref_y']
        z_values = data['ref_z']

        # Combine the data into a list of tuples containing (step, x, y, z) coordinates
        trajectory_points = [(step, x, y, z) for step, x, y, z in zip(steps, x_values, y_values, z_values)]
        return trajectory_points

    def get_trajectory(self):
        """Return the loaded trajectory points."""
        return self.trajectory_points

    def get_point_at_step(self, step):
        """Return the trajectory point corresponding to the given step."""
        if step < len(self.trajectory_points):
            return self.trajectory_points[step][1:]  # Return (x, y, z)
        else:
            # If step is out of bounds, return the last point
            return self.trajectory_points[-1][1:]

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The initial observation of the next episode.
        """
        obs, info = self.env.reset(*args, **kwargs)
        del info["symbolic_model"]
        del info["symbolic_constraints"]
        self.current_step = 0  # Reset the step counter
        obs[-3:] = self.start_waypoint
        self.previous_pos = obs[:3]
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[-3:] = self.get_point_at_step(self.current_step)
        reward = self._compute_reward(obs, reward, terminated, truncated, info)
        # Check if the drone is close enough to the current waypoint
        distance_xy = np.linalg.norm(obs[:2] - obs[-3:-1], ord=2)
        distance_z = np.abs(obs[2] - obs[-1])
        if distance_xy < 0.2 and distance_z < 0.2:
            print(f"Reached waypoint {self.current_step}")
            self.current_step += 1


        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict
    ) -> float:
        """Compute the reward for the current step.

        Args:
            obs: The current observation.
            reward: The reward from the environment.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: Additional information from the environment.

        Returns:
            The computed reward.
        """

        # Compute the distance to the trajectory point at the current step
        next_waypoint = self.get_point_at_step(self.current_step)
        distance_previous = np.linalg.norm(next_waypoint - self.previous_pos, ord=2)
        distance_current = np.linalg.norm(next_waypoint - obs[:3], ord=2)
        trajectory_progress = distance_previous - distance_current
        distance_xy  = np.linalg.norm(obs[:2] - next_waypoint[:2], ord=2)
        distance_z = np.abs(obs[2] - next_waypoint[2])
        #print(f"Distance to z = {distance_z}")
        collision = -1 if terminated and not info["task_completed"] else 0
        reward = trajectory_progress 
      
        self.previous_pos = obs[:3]
        return reward


class RewardWrapper(Wrapper):
    """Wrapper to alter the default reward function from the environment for RL training."""

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)
        self.current_gate = None
        self.current_target = None
        self.previous_pos = None
        self.gate_clear_width = 0.41  # 41 cm clear passage width
        self.total_gate_width = 0.59  # 59 cm total gate width
        self.border_width = 0.09      # 9 cm border width

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The initial observation of the next episode.
        """

        obs, info = self.env.reset(*args, **kwargs)
        del info["symbolic_model"]
        del info["symbolic_constraints"]
        self.current_gate = info["current_gate_id"]
        self.current_target = info["gates_pose"][self.current_gate, :3]
        self.previous_pos = obs[:3]
        #obs[-1] = 0
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """

        #gate_id = info["current_gate_id"]
        #if gate_id > self.current_gate:
        #    self.current_gate = gate_id
        #    self.current_target = info["gates_pose"][self.current_gate, :3]
        #obs[-1] = np.linalg.norm(self.current_target - obs[:3], ord=2)
        # Compute the progress towards the current gate
        #distance_previous = np.linalg.norm(self.current_target - self.previous_pos, ord=2)
        #distance_current = np.linalg.norm(self.current_target - obs[:3], ord=2)
        #trajectory_progress = distance_previous - distance_current
        #obs[-1] = trajectory_progress
        #self.previous_pos = obs[:3]
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(
        self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict
    ) -> float:
        """Compute the reward for the current step.

        Args:
            obs: The current observation.
            reward: The reward from the environment.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: Additional information from the environment.

        Returns:
            The computed reward.
        """
        gate_id = info["current_gate_id"]
        obstacles_in_range = info["obstacles_in_range"]
        obstacle_id = np.where(obstacles_in_range)
        obstacles_list = []

        obstacle_penality = 0
        gate_passed = 0

        if gate_id > self.current_gate:
            self.current_gate = gate_id
            self.current_target = info["gates_pose"][self.current_gate, :3]
            gate_passed = 1


        #if np.any(obstacles_in_range) and terminated:
        #    obstacle_id = np.where(obstacles_in_range)[0][0]
            #euclican distance from the position of the crash to the current target





            #maximal distance from the obstacle to the target

            #print(f"Obstacle penality: {obstacle_penality}")

            #if self._check_bounding_box(obs[:3], self.current_target):
            #    print(f"Passed gate {self.current_gate}")
                #gate_passed = 10
            #    gate_passed = 10
            #    
        collision = -1 if terminated and not info["task_completed"] else 0
        #bodyrate_penalty = 0.01 * np.linalg.norm(obs[9:12], ord=2)
        # Compute the reward
        distance_previous = np.linalg.norm(self.current_target - self.previous_pos, ord=2)
        distance_current = np.linalg.norm(self.current_target - obs[:3], ord=2)
        #get the distance to the nearest obstacle in range
        #penalize the drone if it is close to an obstacle of the obstacle is in range

        if np.any(obstacles_in_range):
            #iterate over the obstacles in range and get the distance to the drone and append it to the list
            for i in obstacle_id[0]:
                obstacles_list.append(np.linalg.norm(obs[:3] - info["obstacles_pose"][i, :3], ord=2))
            obstacles_list = np.array(obstacles_list)
            #get the minimum distance to the drone
            min_distance = np.min(obstacles_list)
            #penalize the drone if it is close to an obstacle
            obstacle_penality = 1 / min_distance
            # distance to the nearest obstacle
            #dist_obstacle = np.linalg.norm(obs[:3] - info["obstacles_pose"][obstacle_id[0][0], :3],axis=1, ord=2)
            #min_distance = np.min(dist_obstacle)    
            #obstacle_penality = 1 / min_distance

    
        trajectory_progress = distance_previous  - distance_current 
        #print(f"Trajectory progress: {trajectory_progress}")
        reward = trajectory_progress + collision 
        # Update the previous position
        self.previous_pos = obs[:3]
        return reward
    
    def _check_bounding_box(self, drone_pose: np.ndarray, current_target: np.ndarray) -> bool:
        """Check if the drone has passed through the current gate without touching the borders."""
        x_g, y_g, _ = current_target
        x_min_clear = x_g - self.gate_clear_width / 2
        x_max_clear = x_g + self.gate_clear_width / 2
        y_min_clear = y_g - self.gate_clear_width / 2
        y_max_clear = y_g + self.gate_clear_width / 2
        in_clear_x_range = x_min_clear <= drone_pose[0] <= x_max_clear
        in_clear_y_range = y_min_clear <= drone_pose[1] <= y_max_clear
        print(f"X: {x_max_clear}, Y: {y_max_clear}")
        passed_gate_clear = in_clear_x_range and in_clear_y_range 
        return passed_gate_clear
    
    def _check_obstacle_collision(self, obs: np.ndarray, info: dict) -> bool:
        
        """Check if the drone near of any obstacle using info["obstacles_in_range"]."""
        obstacles_in_range = info["obstacles_in_range"]
        if np.any(obstacles_in_range):
            print("Obstacle collision detected")
            obstacles = - 0.1
            return True
        return False


class GateWrapper(Wrapper):
    """Wrapper to alter the default reward function from the environment for RL training."""

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)
        self.current_gate = None
        self.current_target = None
        self.previous_pos = None

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The initial observation of the next episode.
        """

        obs, info = self.env.reset(*args, **kwargs)
        del info["symbolic_model"]
        del info["symbolic_constraints"]
        self.current_gate = info["current_gate_id"]
        self.current_target = info["gates_pose"][self.current_gate, :3]
        self.previous_pos = obs[:3]
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        #set yaw to 0
        action[3] = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(
        self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict
    ) -> float:
        """Compute the reward for the current step.

        Args:
            obs: The current observation.
            reward: The reward from the environment.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: Additional information from the environment.

        Returns:
            The computed reward.
        """
        gate_passed =0
        gate_id = info["current_gate_id"]
        if gate_id > self.current_gate:
            self.current_gate = gate_id
            self.current_target = info["gates_pose"][self.current_gate, :3]
            gate_passed = 5

        collision = -1 if terminated and not info["task_completed"] else 0
        bodyrate_penalty = 0.01 * np.linalg.norm(obs[9:12], ord=2)
        r_lap = 10 if terminated and info["task_completed"] else 0
        #if terminated and info["task_completed"]:
        #    print("Agent completed the lap")

        #if terminated and not info["task_completed"]:
        #    print("Agent crashed")
        distance_previous_xy = np.linalg.norm(self.current_target[0:2] - self.previous_pos[:2], ord=2)
        distance_current_xy = np.linalg.norm(self.current_target[0:2] - obs[:2], ord=2)
        gate_progress_xy = distance_previous_xy - distance_current_xy

        distance_previous_z = np.abs(self.current_target[2] - self.previous_pos[2])
        distance_current_z = np.abs(self.current_target[2] - obs[2])
        gate_progress_z = distance_previous_z - distance_current_z
        
        reward = gate_progress_xy + gate_progress_z +  r_lap + collision + gate_passed          
        # Update the previous position
        self.previous_pos = obs[:3]
        return reward