from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.envs.common.fitness_value_extractor import FitnessExtractor
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text
import math

Observation = np.ndarray


class HighwayEnvMA(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                        "vehicles_count":2,
                        "see_behind": True
                    }
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                    "type": "DiscreteMetaAction",
                    }
                },
                "lanes_count": 3,
                "vehicles_count": 0,
                "controlled_vehicles": 2,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,
                "adv_collision_reward": 0,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                "delta" : 0.0025,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        
        rewards_2 = tuple(self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles)
        
        r = []
        adv_vehicle = False
        for rewards in rewards_2:
            
            reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
            
            if self.config["normalize_reward"]:
                if not adv_vehicle:
                    reward = utils.lmap(reward,
                                        [self.config["collision_reward"],
                                        self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                        [0, 1])
                    adv_vehicle = True
                else:
                    reward = utils.lmap(reward,
                                        [self.config["adv_collision_reward"],
                                        self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                        [0, 1])
                
            reward *= rewards['on_road_reward']

            #if rewards['on_road_reward'] != 1.0:
                #print('true')         
            r.append(reward)

        return r

    def _agent_rewards(self, action: Action, vehicle:Vehicle) -> dict[str, float]:
        
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        
        
        #checking whether the vehicle is colliding with a controlled vehicle or not
        #collision reward should be = 0 in case of colliding with controlled vehicle and 1 otherwise
        
        other_vehicle_type = self.config['other_vehicles_type'].split('.')[-1]
        
        #if other_vehicle_type in vehicle.vehicle_in_crash:
            
            #veh_col = float(vehicle.crashed)
        #else:
            #veh_col = 0.0
        
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(vehicle.on_road)
        }

    def _simulate(self, action: tuple[int, int]) -> None:
        """Perform several steps of simulation with constant action, ensuring adversarial agents take valid actions."""

        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        
        # Extract ego and adversarial actions
        ego_action, adv_action = action  # Unpacking the tuple

        ego_vehicle = self.controlled_vehicles[0]
        adv_vehicle = self.controlled_vehicles[1]

        # Modify the adversarial action if it leads to invalid failures
        adv_action = self._safe_adversarial_action(ego_vehicle, adv_vehicle, adv_action)

        # Recreate the action tuple with the safe adversarial action
        action = (ego_action, adv_action)

        for frame in range(frames):
            # Forward action to the vehicles
            if (
                action is not None
                and not self.config["manual_control"]
                and self.steps
                % int(
                    self.config["simulation_frequency"]
                    // self.config["policy_frequency"]
                )
                == 0
            ):
                self.action_type.act(action)

            # Step the simulation
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

    def _safe_adversarial_action(self, ego_vehicle, adv_vehicle, adv_action):
        """Modify adversarial action only if it would result in an invalid failure."""

        # Compute longitudinal and lateral distances
        longitudinal_distance = abs(ego_vehicle.position[0] - adv_vehicle.position[0])  # X-axis
        lateral_distance = abs(ego_vehicle.position[1] - adv_vehicle.position[1])  # Y-axis
        same_lane = ego_vehicle.lane_index[2] == adv_vehicle.lane_index[2]

        # Convert velocities to scalars if necessary
        ego_velocity = np.linalg.norm(ego_vehicle.velocity) if isinstance(ego_vehicle.velocity, np.ndarray) else ego_vehicle.velocity
        adv_velocity = np.linalg.norm(adv_vehicle.velocity) if isinstance(adv_vehicle.velocity, np.ndarray) else adv_vehicle.velocity

        # print(f"LONGITUDINAL DISTANCE: {longitudinal_distance}")
        # print(f"LATERAL: {adv_vehicle.position[1]}" + " " + f"{ego_vehicle.position[1]}")
        # print(f"SAME LANE: {same_lane}")
        # print(f"EGO_V: {ego_velocity}")
        # print(f"ADV_V: {adv_velocity}")

        # # Define discrete action mappings
        # ACTIONS_ALL = {
        #     0: 'LANE_LEFT',
        #     1: 'IDLE',
        #     2: 'LANE_RIGHT',
        #     3: 'FASTER',
        #     4: 'SLOWER'
        # }

        # *** CASE 1: Prevent Rear-End Collisions ***
        # Conditions: same lane, adversary behind ego, higher speed, close distance
        if same_lane and longitudinal_distance < 7 and adv_vehicle.position[0] < ego_vehicle.position[0] and adv_velocity > ego_velocity:
            # print("FORCING 'SLOWER' TO AVOID REAR-END COLLISION")
            return 4  # "SLOWER"

        # *** CASE 2: Prevent Unsafe Lane Changes into Ego ***
        # Condition: different lanes, adversary close in longitudinal direction
        if not same_lane and longitudinal_distance < 7:
            if adv_vehicle.position[1] > ego_vehicle.position[1] and adv_action == 0:
                # print("BLOCKING 'LANE_LEFT' TO AVOID COLLISION")
                return 1  # "IDLE"

            if adv_vehicle.position[1] < ego_vehicle.position[1] and adv_action == 2:
                # print("BLOCKING 'LANE_RIGHT' TO AVOID COLLISION")
                return 1  # "IDLE"

        return adv_action  # Default action if no issues





    def _info(self, obs: Observation, action: Optional[Action] = None, reward: Optional[float]=None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        crashed_list = []
        for vehicle in self.controlled_vehicles:
            crashed_list.append(vehicle.crashed)
        
        crashed = False
        if True in crashed_list:
            crashed = True
        
        
        self.overtaken = False
        time_to_collision_reward = None
        
        if len(obs) == 2:
            
            ego_x_pos = obs[0][0]
        
            adv_x_pos = obs[1][0]
        
            dist_between_agents = adv_x_pos[1] - ego_x_pos[1]
            delta = self.config["delta"]
            #delta = 0.0025
            if dist_between_agents <= delta:
                self.overtaken = True
            
            time_to_collision_reward = self.get_ttc_v2(obs[0], obs[1])
        info = {
            "speed": self.vehicle.speed,
            "crashed": crashed,
            "action": action,
            "crashed_list": crashed_list,
            "overtaken": self.overtaken,
            "time_to_collision_reward": time_to_collision_reward,
        }
        try:
            #info["rewards"] = self._rewards(action)
            rewards_2 = tuple(self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles)
            info["rewards"] = rewards_2
            info["total_rewards"] = reward
        except NotImplementedError:
            pass
        
        return info


    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """

        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action, reward)

        self.fe = FitnessExtractor()

        if terminated:
            adv_reward = 1
        else:
            adv_reward = self.fe.extract_from_env(self)[1]
        # adv_reward = self.fe.extract_from_env(self)[0]

        if self.render_mode == "human":
            self.render()
        
        reward = [info["total_rewards"][0],adv_reward]

        return obs, reward, terminated, truncated, info


    def get_ttc_v2(self, ego_obs, adv_obs, a =1, b =1):
        
        adv_vehicle = adv_obs[0]
        adv_x = adv_vehicle[1]
        adv_y = adv_vehicle[2]
        adv_vx = adv_vehicle[3]
        adv_vy = adv_vehicle[4]
    
        
        ego_vehicle = ego_obs[0]
        ego_x = ego_vehicle[1]
        ego_y = ego_vehicle[2]
        ego_vx = ego_vehicle[3]
        ego_vy = ego_vehicle[4]

        r_x = adv_x - ego_x
        r_y = adv_y - ego_y
        r_vx = adv_vx - ego_vx
        r_vy = adv_vy - ego_vy
        
        reward = self.calculate_reward_1(r_x, r_y, r_vx, r_vy)
        
        
        
        return reward

    def calculate_reward_1(self, r_x, r_y, r_vx, r_vy, alpha = 1, beta = 1):
        
        a=0.1 #previous one
        #a = 0.00001
        if r_vx >= 0:
            r = -r_vx - a 
        elif r_vy != 0:
            r = -abs(r_vy)/3 - a
        else:
            ttc = math.sqrt(r_x**2 + r_y**2)
            r = beta/(alpha + ttc)

        return r



class HighwayEnvFast(HighwayEnvMA):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        # cfg.update(
        #     {
        #         "simulation_frequency": 5,
        #         "lanes_count": 3,
        #         "vehicles_count": 20,
        #         "duration": 30,  # [s]
        #         "ego_spacing": 1.5,
        #     }
        # )
        cfg.update(
            {
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                        "vehicles_count":2,
                        "see_behind": True
                    }
                },
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                    "type": "DiscreteMetaAction",
                    }
                },
                "simulation_frequency": 5,
                "lanes_count": 2,
                "vehicles_count": 0,
                "controlled_vehicles": 2,
                "initial_lane_id": None,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
                "vehicles_density": 1,
                "collision_reward": -1,
                "adv_collision_reward": 0,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                "delta" : 0.0025,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
