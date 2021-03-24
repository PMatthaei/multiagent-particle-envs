import unittest
from unittest.mock import MagicMock

import numpy as np

from multiagent.core import World
from test.mock import mock_agent

N_AGENTS = 6


class WorldAvailableMovementActionsTestCases(unittest.TestCase):
    def setUp(self):
        self.a = mock_agent(id=0, sight_range=20)
        self.b = mock_agent(id=1, tid=0)
        self.c = mock_agent(id=2, tid=0)
        self.d = mock_agent(id=3, tid=1)
        self.e = mock_agent(id=4, tid=1)
        self.f = mock_agent(id=5, tid=1)

        self.world = World(grid_size=10, agents_n=N_AGENTS)
        self.world.agents = [self.a, self.b, self.c, self.d, self.e, self.f]

        self.a_spawn = np.array([20, 20])
        self.b_spawn = np.array([10, 20])
        self.c_spawn = np.array([30, 20])
        self.d_spawn = np.array([20, 10])
        self.e_spawn = np.array([20, 30])
        self.f_spawn = np.array([30, 30])

        self._calculate()

    def _calculate(self):
        self.world.connect(self.a, self.a_spawn)
        self.world.connect(self.b, self.b_spawn)
        self.world.connect(self.c, self.c_spawn)
        self.world.connect(self.d, self.d_spawn)
        self.world.connect(self.e, self.e_spawn)
        self.world.connect(self.f, self.f_spawn)
        self.world._update_alive_status()
        self.world._update_visibility()
        self.world._update_dist_matrix()

    def _replace_agent(self, agent, spawn):
        self.world.connect(agent, spawn)
        self.world._update_alive_status()
        self.world._update_visibility()
        self.world._update_dist_matrix()

    def test_only_enemy_targets_available_if_not_healer_and_in_range(self):
        self.world.calculate_avail_target_actions()

        np.testing.assert_array_equal(self.world.avail_target_actions[self.a.id], [0, 0, 0, 1, 1, 1])

    def test_no_enemy_targets_available_if_not_healer_and_not_in_range(self):
        self.a_spawn = np.array([200, 200])
        self._replace_agent(self.a, self.a_spawn)

        self.world.calculate_avail_target_actions()

        np.testing.assert_array_equal(self.world.avail_target_actions[self.a.id], [0, 0, 0, 0, 0, 0])

    def test_only_mate_targets_available_if_healer_and_in_range(self):
        self.a.has_heal = MagicMock(return_value=True)
        self.world.connect(self.a, self.a_spawn)

        self.world.calculate_avail_target_actions()

        np.testing.assert_array_equal(self.world.avail_target_actions[self.a.id], [0, 1, 1, 0, 0, 0])

    def test_no_mate_targets_available_if_healer_and_not_in_range(self):
        self.a.has_heal = MagicMock(return_value=True)
        self.a_spawn = np.array([200, 200])
        self._replace_agent(self.a, self.a_spawn)

        self.world.calculate_avail_target_actions()

        np.testing.assert_array_equal(self.world.avail_target_actions[self.a.id], [0, 0, 0, 0, 0, 0])

    def test_no_targets_if_no_one_visible(self):
        self.a_spawn = np.array([200, 200])
        self._replace_agent(self.a, self.a_spawn)

        self.world.calculate_avail_target_actions()

        np.testing.assert_array_equal(self.world.avail_target_actions[self.a.id], [0, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
