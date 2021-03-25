import unittest

import numpy as np

from multiagent.core import World
from test.mock import mock_agent

N_AGENTS = 2


class WorldDistancesTestCases(unittest.TestCase):
    def setUp(self):
        self.agent = mock_agent(id=0)
        self.agent2 = mock_agent(id=1, tid=1)
        self.agent_spawn = np.array([0, 0])
        self.agent_spawn2 = np.array([1, 1])

        self.world = World(grid_size=10, teams_n=2, agents_n=N_AGENTS)
        self.world.agents = [self.agent, self.agent2]

        self.world.connect(self.agent, self.agent_spawn)
        self.world.connect(self.agent2, self.agent_spawn2)

    def test_distance_matrix(self):
        self.world._update_dist_matrix()

        np.testing.assert_array_equal(self.world.distances[self.agent.id], [0, np.sqrt(2)])
        np.testing.assert_array_equal(self.world.distances[self.agent2.id], [np.sqrt(2), 0])


if __name__ == '__main__':
    unittest.main()
