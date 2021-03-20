import unittest
from unittest.mock import Mock, MagicMock

import numpy as np

from multiagent.core import World
from test.core.mock import mock_agent

N_AGENTS = 2


class WorldVisibilityTestCases(unittest.TestCase):
    def setUp(self):
        self.agent = mock_agent(id=0)
        self.agent2 = mock_agent(id=1, tid=1)
        self.agent_spawn = np.array([1, 1])
        self.agent_spawn2 = np.array([1, 0])

        self.world = World(grid_size=10, agents_n=N_AGENTS)
        self.world.agents = [self.agent, self.agent2]

        self.world.connect(self.agent, self.agent_spawn)
        self.world.connect(self.agent2, self.agent_spawn2)

    def test_visibility_for_in_range_and_alive(self):
        self.world._update_visibility()

        np.testing.assert_array_equal(self.world.visibility[self.agent.id], [True, True])
        np.testing.assert_array_equal(self.world.visibility[self.agent2.id], [True, True])

    def test_no_visibility_for_in_range_and_dead(self):
        self.agent2.is_alive = MagicMock(return_value=False)  # declare dead

        self.world.connect(self.agent2, self.agent_spawn2)

        self.world._update_visibility()

        np.testing.assert_array_equal(self.world.visibility[self.agent.id], [True, False])
        np.testing.assert_array_equal(self.world.visibility[self.agent2.id], [False, False])

    def test_visibility_for_not_in_range_and_alive(self):
        self.world.connect(self.agent2, np.array([3, 3]))  # move out of range

        self.world._update_visibility()

        np.testing.assert_array_equal(self.world.visibility[self.agent.id], [True, False])
        np.testing.assert_array_equal(self.world.visibility[self.agent2.id], [False, True])

    def test_no_visibility_for_not_in_range_and_dead(self):
        self.agent2.is_alive = MagicMock(return_value=False)  # declare dead
        self.world.connect(self.agent2, np.array([3, 3]))  # move out of range

        self.world._update_visibility()

        np.testing.assert_array_equal(self.world.visibility[self.agent.id], [True, False])
        np.testing.assert_array_equal(self.world.visibility[self.agent2.id], [False, False])


if __name__ == '__main__':
    unittest.main()
