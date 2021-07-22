import unittest
from unittest.mock import MagicMock

import numpy as np

from maenv.core import World
from test.mock import mock_agent

N_AGENTS = 6


class WorldAvailableTargetActionsTestCases(unittest.TestCase):
    def setUp(self):
        self.world = World(grid_size=10, n_teams=2, n_agents=4)

        self.world.attack_target_mask = np.array([
            [0, 0, 1, 1],  # attacker
            [0, 0, 0, 0],  # healer
            [0, 0, 1, 1],  # attacker
            [0, 0, 0, 0]  # healer
        ])
        self.world.heal_target_mask = np.array([
            [0, 0, 0, 0],  # attacker
            [1, 1, 0, 0],  # healer
            [0, 0, 0, 0],  # attacker
            [1, 1, 0, 0]  # healer
        ])
        self.world.reachability = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ])
        self.world.alive = np.array([1, 1, 1, 1])

    def test_no_target_action_available_if_alive_and_no_enemy_reachable_and_attacker(self):
        self.world.calculate_avail_target_actions()
        np.testing.assert_array_equal(self.world.avail_target_actions[0], [0, 0, 0, 0])

    def test_no_target_action_available_if_alive_and_only_enemy_reachable_and_healer(self):
        self.world.reachability[1] = [0, 0, 1, 1]
        self.world.calculate_avail_target_actions()
        np.testing.assert_array_equal(self.world.avail_target_actions[1], [0, 0, 0, 0])

    def test_mate_target_action_available_if_alive_and_mate_reachable_and_healer(self):
        self.world.calculate_avail_target_actions()
        np.testing.assert_array_equal(self.world.avail_target_actions[1], [1, 0, 0, 0])

    def test_no_target_action_if_dead(self):
        self.world.alive = np.array([0, 1, 1, 1])
        self.world.calculate_avail_target_actions()
        np.testing.assert_array_equal(self.world.avail_target_actions[0], [0, 0, 0, 0])

    def test_only_enemy_targets_available_if_alive_and_all_enemy_reachable_and_attacker(self):
        self.world.reachability[0] = [0, 0, 1, 1]
        self.world.calculate_avail_target_actions()
        np.testing.assert_array_equal(self.world.avail_target_actions[0], [0, 0, 1, 1])


if __name__ == '__main__':
    unittest.main()
