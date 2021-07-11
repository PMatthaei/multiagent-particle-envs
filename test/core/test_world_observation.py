import unittest
from unittest.mock import MagicMock

import numpy as np

from maenv.core import World
from test.mock import mock_agent

N_AGENTS = 2
HEALTH = 40.0
SIGHT_RANGE_A = 20.0
SIGHT_RANGE_B = 10.0
DISTANCE_TO_B = 10.0
DISTANCE_TO_A = 10.0


class WorldObservationTestCases(unittest.TestCase):
    def setUp(self):
        self.grid_size = 10
        self.world = World(grid_size=self.grid_size, n_teams=2, n_agents=N_AGENTS)

        self.world.visibility = np.array([
            [1, 1],
            [1, 1]
        ])
        self.world.positions = np.array([
            [0, 0],
            [10, 10]
        ])
        self.world.sight_ranges = np.array([SIGHT_RANGE_A, SIGHT_RANGE_B])
        self.world.distances = np.array([
            [0, DISTANCE_TO_B],
            [DISTANCE_TO_A, 0]
        ])
        self.world.health = np.array([HEALTH] * N_AGENTS)
        self.world.max_health = np.array([HEALTH] * N_AGENTS)
        self.world.unit_bits_obs = np.array([
            [0, 0, 1],
            [0, 0, 1]
        ])
        self.world.agents_n = N_AGENTS

    def test_a_observes_b_if_visible_and_alive(self):
        #   vis     health    rel x-pos   rel y-pos  rel dist  unit bits
        self.world._calculate_obs()
        a_obs_of_b = np.array(
            [1.0, 1.0, 10.0 / SIGHT_RANGE_A, 10.0 / SIGHT_RANGE_A, DISTANCE_TO_B / SIGHT_RANGE_A, 0, 0, 1])
        np.testing.assert_array_equal(self.world.obs[0][1], a_obs_of_b)  # agent at 0 observes agent at 1

    def test_b_observes_a_with_different_sight_range_if_visible_and_alive(self):
        self.world._calculate_obs()
        b_obs_of_a = np.array(
            [1.0, 1.0, -10.0 / SIGHT_RANGE_B, -10.0 / SIGHT_RANGE_B, DISTANCE_TO_A / SIGHT_RANGE_B, 0, 0, 1])
        np.testing.assert_almost_equal(self.world.obs[1][0], b_obs_of_a)

    def test_a_cannot_observe_b_if_b_not_visible(self):
        self.world.visibility = np.array([
            [1, 0],
            [1, 1]
        ])
        self.world._calculate_obs()
        a_obs_of_b = np.zeros((self.world.obs_dims,))
        np.testing.assert_array_equal(self.world.obs[0][1], a_obs_of_b)


if __name__ == '__main__':
    unittest.main()
