import unittest

import numpy as np

from maenv.core import World
from test.mock import mock_agent
import scipy.spatial.ckdtree
import scipy.spatial.distance
N_AGENTS = 5


class WorldAvailableMovementActionsTestCases(unittest.TestCase):
    def setUp(self):
        self.a = mock_agent(id=0)
        self.b = mock_agent(id=1, tid=0)
        self.c = mock_agent(id=2, tid=1)
        self.d = mock_agent(id=3, tid=1)
        self.e = mock_agent(id=4, tid=1)

        self.world = World(grid_size=10, n_teams=2, n_agents=N_AGENTS)
        self.world.agents = [self.a, self.b, self.c, self.d, self.e]

        self.a_spawn = np.array([100, 100])
        self.b_spawn = np.array([10, 20])
        self.c_spawn = np.array([30, 20])
        self.d_spawn = np.array([20, 10])
        self.e_spawn = np.array([20, 30])

        self._calculate()

    def _calculate(self):
        self.world.connect(self.a, self.a_spawn)
        self.world.connect(self.b, self.b_spawn)
        self.world.connect(self.c, self.c_spawn)
        self.world.connect(self.d, self.d_spawn)
        self.world.connect(self.e, self.e_spawn)
        self.world.kd_tree = scipy.spatial.cKDTree(data=self.world.positions)
        self.world._update_alive_status()
        self.world._update_visibility()
        self.world._update_dist_matrix()

    def test_all_free(self):
        self.world.calculate_avail_movements_actions()

        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [1, 1, 1, 1])

    def test_all_blocked(self):
        self.a_spawn = np.array([20, 20])
        self._calculate()

        self.world.calculate_avail_movements_actions()

        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [0, 0, 0, 0])

    def test_blocked_west_by_other_agent(self):
        self.b_spawn = np.array([90, 100])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [0, 1, 1, 1])

    def test_blocked_east_by_other_agent(self):
        self.b_spawn = np.array([110, 100])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [1, 0, 1, 1])

    def test_blocked_south_by_other_agent(self):
        self.b_spawn = np.array([100, 90])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [1, 1, 1, 0])

    def test_blocked_north_by_other_agent(self):
        self.b_spawn = np.array([100, 110])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [1, 1, 0, 1])

    def test_blocked_west_by_bound(self):
        self.a_spawn = np.array([0, 10])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [0, 1, 1, 1])

    def test_blocked_east_by_bound(self):
        self.a_spawn = np.array([self.world.bounds[0], 10])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [1, 0, 1, 1])

    def test_blocked_south_by_bound(self):
        self.a_spawn = np.array([10, 0])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [1, 1, 1, 0])

    def test_blocked_north_by_bound(self):
        self.a_spawn = np.array([10, self.world.bounds[1]])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [1, 1, 0, 1])

    def test_blocked_west_and_south_by_bound(self):
        self.a_spawn = np.array([0, 0])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [0, 1, 1, 0])

    def test_blocked_east_and_south_by_bound(self):
        self.a_spawn = np.array([self.world.bounds[0], 0])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [1, 0, 1, 0])

    def test_blocked_east_and_north_by_bound(self):
        self.a_spawn = np.array([self.world.bounds[0], self.world.bounds[1]])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [1, 0, 0, 1])

    def test_blocked_west_and_north_by_bound(self):
        self.a_spawn = np.array([0, self.world.bounds[1]])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.a.id], [0, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
