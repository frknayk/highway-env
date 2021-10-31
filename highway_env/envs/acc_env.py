import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle

class AccTwoLanesEnv(AbstractEnv):

    """
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 2,
            "vehicles_count": 1,
            "controlled_vehicles": 1,
            "initial_lane_id": 0,
            "duration": 40,  # [s]
            "ego_spacing": 2,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self, length=800) -> None:
        """Create a road composed of straight adjacent lanes."""
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

    def _spawn_ego_vehicle(self) -> None:
        """
        Spawn ego vehicle - SUT: System under test
        """
        self.vehicle_ego = self.action_type.vehicle_class(self.road,
                                                     self.road.network.get_lane(("a", "b", 1)).position(0, 0),
                                                     speed=5)
        self.road.vehicles.append(self.vehicle_ego)

    def _spawn_agents(self) -> None:
        """
        Spawn agent vehicles controlled by AI
        """
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicles_type(self.road,
                        position=self.road.network.get_lane(("a", "b", 1))
                        .position(70+40*i + 10*self.np_random.randn(), 0),
                        heading=self.road.network.get_lane(("a", "b", 1)).heading_at(70+40*i),
                        speed=24 + 2*self.np_random.randn(),
                        enable_lane_change=False)

        # for i in range(3):
        #     self.road.vehicles.append(
        #         vehicles_type(self.road,
        #                       position=self.road.network.get_lane(("a", "b", 1))
        #                       .position(70+40*i + 10*self.np_random.randn(), 0),
        #                       heading=self.road.network.get_lane(("a", "b", 1)).heading_at(70+40*i),
        #                       speed=24 + 2*self.np_random.randn(),
        #                       enable_lane_change=False)
        #     )
        # for i in range(2):
        #     v = vehicles_type(self.road,
        #                       position=self.road.network.get_lane(("b", "a", 0))
        #                       .position(200+100*i + 10*self.np_random.randn(), 0),
        #                       heading=self.road.network.get_lane(("b", "a", 0)).heading_at(200+100*i),
        #                       speed=20 + 5*self.np_random.randn(),
        #                       enable_lane_change=False)
        #     v.target_lane_index = ("b", "a", 0)
        #     self.road.vehicles.append(v)

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
        return self.vehicle_ego.crashed or self.steps >= self.config["duration"]

    def _cost(self, action: int) -> float:
        """
        The cost signal is the occurrence of collision.
        """
        return float(self.vehicle_ego.crashed)


register(
    id='acc-two-lanes-v0',
    entry_point='highway_env.envs:AccTwoLanesEnv',
    max_episode_steps=15
)
