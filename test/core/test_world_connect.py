import unittest
from unittest.mock import MagicMock

import numpy as np

from maenv.core import World
from test.mock import mock_agent

N_AGENTS = 2


class WorldInitializationTestCases(unittest.TestCase):
    def setUp(self):
        self.agent = mock_agent(id=0)
        self.agent2 = mock_agent(id=1, tid=1)
        self.agent_spawn = np.array([1, 1])

        self.grid_size = 10
        self.world = World(grid_size=self.grid_size, n_teams=2, n_agents=N_AGENTS)
        self.world.agents = [self.agent, self.agent2]

    def test_connects_as_alive(self):
        self.world.connect(self.agent, self.agent_spawn)

        np.testing.assert_array_equal(self.world.positions[0], [1, 1])

    def test_connects_as_dead(self):
        self.agent.is_alive = MagicMock(return_value=False)
        self.world.connect(self.agent, None)

        np.testing.assert_array_equal(self.world.alive[0], 0)

    def test_connects_health(self):
        self.world.connect(self.agent, None)

        np.testing.assert_array_equal(self.world.health[0], 100)

    def test_connects_max_health(self):
        self.world.connect(self.agent, None)

        np.testing.assert_array_equal(self.world.max_health[0], 100)

    def test_connects_range(self):
        self.world.connect(self.agent, None)

        np.testing.assert_array_equal(self.world.sight_ranges[0], self.agent.sight_range * self.grid_size)

    def test_connects_attack_target_mask(self):
        self.world.connect(self.agent, None)
        self.world.connect(self.agent2, None)

        np.testing.assert_array_equal(self.world.attack_target_mask[0], [False, True])
        np.testing.assert_array_equal(self.world.attack_target_mask[1], [True, False])


if __name__ == '__main__':
    unittest.main()
