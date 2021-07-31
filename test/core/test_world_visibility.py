import unittest
from unittest.mock import MagicMock

import numpy as np
import scipy.spatial.ckdtree
import scipy.spatial.distance

from maenv.core import World
from test.mock import mock_agent

N_AGENTS = 2


class WorldVisibilityTestCases(unittest.TestCase):
    def setUp(self):
        self.aid = 0
        self.bid = 1

        attack_range_a = 1
        attack_range_b = 1

        self.grid_size = 10
        self.world = World(grid_size=self.grid_size, n_teams=2, n_agents=N_AGENTS)
        self.world.positions = np.array([[10, 10], [10, 0]])
        self.world.sight_ranges = np.array([attack_range_a * self.grid_size, attack_range_b * self.grid_size])
        self.world.alive = np.array([1, 1])
        self.world.kd_tree = scipy.spatial.cKDTree(data=self.world.positions)

    def test_a_can_see_b(self):
        self.world._update_visibility()
        np.testing.assert_array_equal([1, 1], self.world.visible[self.aid])

    def test_a_can_not_see_b_because_out_of_range(self):
        self.world.positions[self.bid] = [100, 0]  # move b out of range
        self.world.kd_tree = scipy.spatial.cKDTree(data=self.world.positions)
        self.world._update_visibility()
        np.testing.assert_array_equal([1, 0], self.world.visible[self.aid])

    def test_a_can_not_see_b_because_not_alive(self):
        self.world.alive[self.bid] = 0  # b dead
        self.world._update_visibility()
        np.testing.assert_array_equal([1, 0], self.world.visible[self.aid])

    def test_a_can_not_see_b_because_not_alive_and_out_of_range(self):
        self.world.alive[self.bid] = 0  # b dead
        self.world.positions[self.bid] = [100, 0]  # move b out of range
        self.world._update_visibility()
        np.testing.assert_array_equal([1, 0], self.world.visible[self.aid])


if __name__ == '__main__':
    unittest.main()
