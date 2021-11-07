import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle

class AccTwoLanesEnv(AbstractEnv):

    """
    Adaptive stress testing environment for 
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            # "observation": {
            #     "type": "Kinematics",
            #     "normalize": False,
            #     "absolute": True
            # },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "normalize": False,
                    "absolute": True
                }
            },
            "action": {
                "type": "MultiAgentAction",
                 "action_config": {
                     "type": "ContinuousAction",
                     "lateral": False,
                     "longitudinal": True
                 }
            },
            # "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "lanes_count": 2,
            "vehicles_count": 0,
            "controlled_vehicles": 2,
            "duration": 4000,  # [s]
            "ego_spacing": 2,
            "road_length" : 1000,
            "show_trajectories": True,
            "lane_init_id_ego": 0,
            "lane_init_id_agent": 0,
            "pos_init_ego": 30,
            "pos_init_agent": 50,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()


    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self._create_ego_vehicle()
        self._create_agent_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        length = self.config["road_length"]
        net = RoadNetwork()
        # Lanes
        net.add_lane("a", "b", StraightLane([0, 0], [length, 0],
                                            line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED)))
        net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH], [length, StraightLane.DEFAULT_WIDTH],
                                            line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)))
        net.add_lane("b", "a", StraightLane([length, 0], [0, 0],
                                            line_types=(LineType.NONE, LineType.NONE)))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _create_ego_vehicle(self) -> None:
        road = self.road
        lane_ego = road.network.get_lane(("a", "b", self.config['lane_init_id_ego']))
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     lane_ego.position(self.config['pos_init_ego'], 0),
                                                     speed=0)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _create_agent_vehicles(self) -> None:
        road = self.road
        lane_agent = road.network.get_lane(("a", "b", self.config['lane_init_id_agent']))
        vehicle_agent = self.action_type.vehicle_class(road,
                                                     position=lane_agent.position(self.config['pos_init_agent'], 0),
                                                     heading=lane_agent.heading_at(0),
                                                     speed=0)
        self.road.vehicles.append(vehicle_agent)

    def _reward(self, action: int) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        return 0.0

    def _is_terminal(self) -> bool:
        """
        The episode is over if the ego vehicle crashed or the time is out.
        """
        # is_terminal =  self.vehicle.crashed or self.steps >= self.config["duration"]
        is_terminal =  self.vehicle.crashed
        print("steps: {0} / duration : {1} / : is_terminal {2}".format(self.steps,
                self.config["duration"],is_terminal))
        return is_terminal

    def _cost(self, action: int) -> float:
        """
        The cost signal is the occurrence of collision.
        """
        return float(self.vehicle.crashed)


register(
    id='acc-two-lanes-v0',
    entry_point='highway_env.envs:AccTwoLanesEnv'
)
