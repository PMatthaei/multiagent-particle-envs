import unittest
from unittest.mock import MagicMock

import numpy as np

from multiagent.core import World
from test.mock import mock_agent

N_AGENTS = 2


class WorldInitializationTestCases(unittest.TestCase):
    def setUp(self):
        self.agent = mock_agent(id=0)
        self.agent2 = mock_agent(id=1, tid=1)
        self.agent_spawn = np.array([1, 1])

        self.world = World(grid_size=10, teams_n=2, agents_n=N_AGENTS)
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

        np.testing.assert_array_equal(self.world.ranges[0], 2)

    def test_connects_attack_target_mask(self):
        self.world.connect(self.agent, None)
        self.world.connect(self.agent2, None)

        np.testing.assert_array_equal(self.world.attack_target_mask[0], [False, True])
        np.testing.assert_array_equal(self.world.attack_target_mask[1], [True, False])


if __name__ == '__main__':
    unittest.main()
