import unittest
from unittest.mock import Mock, MagicMock

import numpy as np

from multiagent.core import World
from test.core.mock import mock_agent

N_AGENTS = 5


class WorldAvailableMovementActionsTestCases(unittest.TestCase):
    def setUp(self):
        self.agent = mock_agent(id=0)
        self.agent2 = mock_agent(id=1, tid=0)
        self.agent3 = mock_agent(id=2, tid=1)
        self.agent4 = mock_agent(id=3, tid=1)
        self.agent5 = mock_agent(id=4, tid=1)

        self.world = World(grid_size=10, agents_n=N_AGENTS)
        self.world.agents = [self.agent, self.agent2, self.agent3, self.agent4, self.agent5]

        self.agent_spawn = np.array([100, 100])
        self.agent_spawn2 = np.array([10, 20])
        self.agent_spawn3 = np.array([30, 20])
        self.agent_spawn4 = np.array([20, 10])
        self.agent_spawn5 = np.array([20, 30])

        self._calculate()

    def _calculate(self):
        self.world.connect(self.agent, self.agent_spawn)
        self.world.connect(self.agent2, self.agent_spawn2)
        self.world.connect(self.agent3, self.agent_spawn3)
        self.world.connect(self.agent4, self.agent_spawn4)
        self.world.connect(self.agent5, self.agent_spawn5)
        self.world._update_alive_status()
        self.world._update_visibility()
        self.world._update_dist_matrix()

    def test_all_free(self):
        self.world.calculate_avail_movements_actions()

        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [1, 1, 1, 1])

    def test_all_blocked(self):
        self.agent_spawn = np.array([20, 20])
        self._calculate()

        self.world.calculate_avail_movements_actions()

        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [0, 0, 0, 0])

    def test_blocked_west_by_other_agent(self):
        self.agent_spawn2 = np.array([90, 100])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [0, 1, 1, 1])

    def test_blocked_east_by_other_agent(self):
        self.agent_spawn2 = np.array([110, 100])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [1, 0, 1, 1])

    def test_blocked_north_by_other_agent(self):
        self.agent_spawn2 = np.array([100, 90])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [1, 1, 0, 1])

    def test_blocked_south_by_other_agent(self):
        self.agent_spawn2 = np.array([100, 110])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [1, 1, 1, 0])

    def test_blocked_west_by_bound(self):
        self.agent_spawn = np.array([0, 10])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [0, 1, 1, 1])

    def test_blocked_east_by_bound(self):
        self.agent_spawn = np.array([self.world.bounds[0], 10])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [1, 0, 1, 1])

    def test_blocked_north_by_bound(self):
        self.agent_spawn = np.array([self.world.bounds[0] - 10, 0])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [1, 1, 0, 1])

    def test_blocked_south_by_bound(self):
        self.agent_spawn = np.array([self.world.bounds[0] - 10, self.world.bounds[1]])
        self._calculate()

        self.world.calculate_avail_movements_actions()
        np.testing.assert_array_equal(self.world.avail_movement_actions[self.agent.id], [1, 1, 1, 0])


if __name__ == '__main__':
    unittest.main()
