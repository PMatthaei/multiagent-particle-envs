import unittest
from unittest.mock import MagicMock

import numpy as np

from multiagent.core import World
from test.mock import mock_agent

N_AGENTS = 2


class WorldObservationTestCases(unittest.TestCase):
    def setUp(self):
        self.a = mock_agent(id=0)
        self.b = mock_agent(id=1, tid=1)
        self.b.sight_range = 3
        self.a_spawn = np.array([0, 0])
        self.b_spawn = np.array([1, 1])

        self.world = World(grid_size=10, teams_n=2, agents_n=N_AGENTS)
        self.world.agents = [self.a, self.b]

        self.world.connect(self.a, self.a_spawn)
        self.world.connect(self.b, self.b_spawn)
        self.world._update_visibility()

        self.world._update_dist_matrix()

        self.world._calculate_obs()

    def test_a_observes_b_if_visible_and_alive(self):
        #   vis     health    rel x-pos   rel y-pos  rel dist  unit bits
        r = self.a.sight_range
        a_obs_of_b = np.array([1.0, 1.0, 1.0 / r, 1.0 / r, np.sqrt(2) / r, 0, 0, 1])
        np.testing.assert_array_equal(self.world.obs[self.a.id][self.b.id], a_obs_of_b)

    def test_b_observes_a_with_different_sight_range_if_visible_and_alive(self):
        r = self.b.sight_range
        b_obs_of_a = np.array([1.0, 1.0, -1.0 / r, -1.0 / r, np.sqrt(2) / r, 0, 0, 1])
        np.testing.assert_array_equal(self.world.obs[self.b.id][self.a.id], b_obs_of_a)

    def test_a_cannot_observe_b_if_b_dead(self):
        self.b.is_alive = MagicMock(return_value=False)
        self.world.connect(self.b, self.b_spawn)
        self.world._update_visibility()
        self.world._update_dist_matrix()
        self.world._calculate_obs()

        a_obs_of_b = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(self.world.obs[self.a.id][self.b.id], a_obs_of_b)

    def test_a_cannot_observe_b_if_b_not_visible_and_alive(self):
        self.b_spawn = np.array([4, 0])
        self.world.connect(self.b, self.b_spawn)
        self.world._update_visibility()
        self.world._update_dist_matrix()
        self.world._calculate_obs()

        a_obs_of_b = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(self.world.obs[self.a.id][self.b.id], a_obs_of_b)


if __name__ == '__main__':
    unittest.main()
