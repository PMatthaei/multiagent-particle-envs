import unittest
from unittest.mock import MagicMock

import numpy as np

from maenv.core import World
from test.mock import mock_agent

N_AGENTS = 6


class WorldAvailableMovementActionsTestCases(unittest.TestCase):
    def setUp(self):
        self.world = World(grid_size=10, n_teams=2, n_agents=4)

        self.world.attack_target_mask = np.array([1, 0, 1, 0])
        self.world.visibility = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        self.world.heal_target_mask = np.array([1, 0, 1, 0])
        self.world.alive = np.array([1, 1, 1, 1])

    def test_only_enemy_targets_available_if_not_healer_and_in_range(self):
        self.world.calculate_avail_target_actions()
        np.testing.assert_array_equal(self.world.avail_target_actions[0], [1, 1, 1, 1])

    # def test_no_enemy_targets_available_if_not_healer_and_not_in_range(self):
    #     self.a_spawn = np.array([200, 200])
    #     self._replace_agent(self.a, self.a_spawn)
    #
    #     self.world.calculate_avail_target_actions()
    #
    #     np.testing.assert_array_equal(self.world.avail_target_actions[self.a.id], [0, 0, 0, 0, 0, 0])
    #
    # def test_only_team_targets_available_if_healer_and_in_range(self):
    #     self.a.has_heal = MagicMock(return_value=True)
    #     self.world.connect(self.a, self.a_spawn)
    #
    #     self.world.calculate_avail_target_actions()
    #
    #     np.testing.assert_array_equal(self.world.avail_target_actions[self.a.id], [0, 1, 1, 0, 0, 0])
    #
    # def test_no_mate_targets_available_if_healer_and_not_in_range(self):
    #     self.a.has_heal = MagicMock(return_value=True)
    #     self.a_spawn = np.array([200, 200])
    #     self._replace_agent(self.a, self.a_spawn)
    #
    #     self.world.calculate_avail_target_actions()
    #
    #     np.testing.assert_array_equal(self.world.avail_target_actions[self.a.id], [0, 0, 0, 0, 0, 0])
    #
    # def test_no_targets_if_no_one_visible(self):
    #     self.a_spawn = np.array([200, 200])
    #     self._replace_agent(self.a, self.a_spawn)
    #
    #     self.world.calculate_avail_target_actions()
    #
    #     np.testing.assert_array_equal(self.world.avail_target_actions[self.a.id], [0, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
