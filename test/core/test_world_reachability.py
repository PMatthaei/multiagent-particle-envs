import unittest
import scipy.spatial.ckdtree
import scipy.spatial.distance
import numpy as np

from maenv.core import World
from test.mock import mock_agent

N_AGENTS = 2


class WorldReachabilityTestCases(unittest.TestCase):
    def setUp(self):
        self.a = mock_agent(id=0)
        self.b = mock_agent(id=1, tid=1, sight_range=3)
        attack_range_a = 1
        attack_range_b = 1

        self.grid_size = 10
        self.world = World(grid_size=self.grid_size, n_teams=2, n_agents=N_AGENTS)
        self.world.positions = np.array([[10, 10], [10, 0]])
        self.world.attack_ranges = np.array([attack_range_a * self.grid_size, attack_range_b * self.grid_size])
        self.world.alive = np.array([1, 1])
        self.world.kd_tree = scipy.spatial.cKDTree(data=self.world.positions)

    def test_a_can_attack_b(self):
        self.world._update_reachability()
        result = self.world.can_attack(self.a, self.b)
        self.assertEqual(True, result)

    def test_a_can_not_attack_b_because_out_of_range(self):
        self.world.positions[1] = [100, 0]  # move b out of range
        self.world.kd_tree = scipy.spatial.cKDTree(data=self.world.positions)
        self.world._update_reachability()
        result = self.world.can_attack(self.a, self.b)
        self.assertEqual(False, result)

    def test_a_can_not_attack_b_because_not_alive(self):
        self.world.alive[1] = 0  # b dead
        self.world._update_reachability()
        result = self.world.can_attack(self.a, self.b)
        self.assertEqual(False, result)

    def test_a_can_not_attack_b_because_not_alive_and_out_of_range(self):
        self.world.alive[1] = 0  # b dead
        self.world.positions[1] = [100, 0]  # move b out of range
        self.world.kd_tree = scipy.spatial.cKDTree(data=self.world.positions)
        self.world._update_reachability()
        result = self.world.can_attack(self.a, self.b)
        self.assertEqual(False, result)


if __name__ == '__main__':
    unittest.main()
