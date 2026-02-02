import math
import numpy as np
import gymnasium as gym
import highway_env

class FitnessExtractor:
    def __init__(self):
        self.collision_threshold = 0 # highway has collsion detection 
        self.REWARD_ON_CONTROLLED = False

    def calculate_distance(self, ego, other):
        """Calculate Euclidean distance between ego and another vehicle."""
        return np.linalg.norm(np.array(ego.position) - np.array(other.position))

    def judge_same_lane(self, ego, other):
        """Determines if two vehicles are on the same lane using lane ID."""

        if ego.lane_index[0] != other.lane_index[0]:
            return False

        return ego.lane_index[2] == other.lane_index[2]  

    def is_moving_towards(self, vehicle, target_x, target_y):
        """Check if vehicle is moving toward a given point."""
        pos = np.array(vehicle.position)
        vel = np.array(vehicle.velocity)
        direction = np.array([target_x, target_y]) - pos

        if np.linalg.norm(direction) == 0 or np.linalg.norm(vel) == 0:
            return False

        direction /= np.linalg.norm(direction)
        vel /= np.linalg.norm(vel)

        return np.dot(direction, vel) > 0  

    def get_line(self, vehicle):
        """Returns the slope and intercept of the trajectory line."""
        x, y = vehicle.position
        vx, vy = vehicle.velocity
        vx = vx if vx != 0 else 0.0001  
        slope = vy / vx
        intercept = y - slope * x
        return slope, intercept

    def calculate_angle_tan(self, k1, k2):
        """Calculate angle between two trajectories using tangent function."""
        if k1 == k2:
            k2 = k2 - 0.0001  
        tan_theta = abs((k1 - k2) / (1 + k1 * k2))
        return np.arctan(tan_theta)

    def calculate_collision_probability(self, safe_distance, current_distance):
        """Calculate probability of collision based on safe distance."""
        if current_distance >= safe_distance:
            return 0
        return (safe_distance - current_distance) / safe_distance

    def get_collision_probability(self, ego_vehicle, vehicle):
        """Compute probability of collision based on vehicle alignment, trajectory, and distance."""

        ego_speed = np.linalg.norm(ego_vehicle.velocity) or 0.0001
        ego_slope, ego_intercept = self.get_line(ego_vehicle)
        ego_deceleration = 6  

        loProC_list, laProC_list = [0], [0]  

        vehicle_speed = np.linalg.norm(vehicle.velocity) or 0.0001
        distance = self.calculate_distance(ego_vehicle, vehicle)

        vehicle_slope, vehicle_intercept = self.get_line(vehicle)

        # collision_point_x, collision_point_y = (vehicle_intercept - ego_intercept) / (ego_slope - vehicle_slope), \
        #                                         (ego_slope * vehicle_intercept - vehicle_slope * ego_intercept) / (ego_slope - vehicle_slope)

        same_lane = self.judge_same_lane(ego_vehicle, vehicle)

        if same_lane:
            # print("SAME LANE DETECTED")
            vehicle_deceleration = 6  
            loSD = 1 / 2 * (
                    abs(pow(ego_speed, 2) / ego_deceleration - pow(vehicle_speed, 2) / vehicle_deceleration)) + 5

            # Check if vehicle is moving toward collision point
            # if not self.is_moving_towards(ego_vehicle, collision_point_x, collision_point_y):
            #     loProC = 0
            # elif not (self.is_moving_towards(ego_vehicle, vehicle.position[0], vehicle.position[1]) or
            #             self.is_moving_towards(vehicle, ego_vehicle.position[0], ego_vehicle.position[1])):
            #     loProC = 0
            # else:
            loProC = self.calculate_collision_probability(loSD, distance)

            loProC_list.append(loProC)

        else:
            # print("DIFFERENT LANE")
            theta = self.calculate_angle_tan(ego_slope, vehicle_slope)
            laSD = pow(ego_speed * math.sin(theta), 2) / (ego_deceleration * math.sin(theta)) + 5

            # if not self.is_moving_towards(ego_vehicle, collision_point_x, collision_point_y):
            #     laProC = 0
            # else:
            laProC = self.calculate_collision_probability(laSD, distance)

            laProC_list.append(laProC)

        loProC_dt = max(loProC_list)
        laProC_dt = max(laProC_list)
        proC_dt = max(loProC_dt, laProC_dt) + (1 - max(loProC_dt, laProC_dt)) * min(loProC_dt, laProC_dt)

        return proC_dt

    def extract_from_env(self, env):
        """Extracts collision-related metrics from the HighwayEnv environment."""
        ego_vehicle = env.controlled_vehicles[0]
        adv_vehicle = env.controlled_vehicles[1]

        min_distance = self.calculate_distance(ego_vehicle, adv_vehicle)

        # print("DIST: " + str(min_distance))

        if min_distance <= self.collision_threshold:
            # print("THRESHOLD")
            collision_probability = 1
            TTC = 0
        else:
            collision_probability = round(float(self.get_collision_probability(ego_vehicle, adv_vehicle)), 3)
            TTC = 100000  

        distance_reward = -math.log(min_distance / 40) if 0 <= min_distance <= 40 else 0

        return [distance_reward, collision_probability]
