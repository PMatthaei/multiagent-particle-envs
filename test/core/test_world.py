import unittest
from unittest.mock import Mock, MagicMock

import numpy as np

from multiagent.core import World

N_AGENTS = 2


class WorldTestCases(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.agent.id = 0
        self.agent.state.pos = np.array([1, 1])
        self.agent.is_alive = MagicMock(return_value=True)
        self.agent.sight_range = 2

        self.agent2 = Mock()
        self.agent2.id = 1
        self.agent2.state.pos = np.array([0, 0])
        self.agent2.is_alive = MagicMock(return_value=True)
        self.agent2.sight_range = 2

        self.world = World(grid_size=10, agents_n=N_AGENTS)

    def test_occupied_position_matrix_init(self):
        oc_shape = self.world.occupied_positions.shape
        self.assertEqual(oc_shape, (N_AGENTS, 3))

    def test_distance_matrix_init(self):
        ds_shape = self.world.distance_matrix.shape
        self.assertEqual(ds_shape, (N_AGENTS, N_AGENTS))

    def test_visibility_matrix_init(self):
        vs_shape = self.world.visibility_matrix.shape
        self.assertEqual(vs_shape, (N_AGENTS, N_AGENTS))

    def test_occupies_as_alive(self):
        self.world.occupy_pos(self.agent)

        np.testing.assert_array_equal(self.world.occupied_positions[0], [1, 1, 1])

    def test_occupies_as_dead(self):
        self.agent.is_alive = MagicMock(return_value=False)
        self.world.occupy_pos(self.agent)

        np.testing.assert_array_equal(self.world.occupied_positions[0], [1, 1, 0])

    def test_visibility_for_in_range_and_alive(self):
        self.world.occupy_pos(self.agent)
        self.world.occupy_pos(self.agent2)

        self.world._calculate_visibility(self.agent)
        self.world._calculate_visibility(self.agent2)

        np.testing.assert_array_equal(self.world.visibility_matrix[self.agent.id], [True, True])
        np.testing.assert_array_equal(self.world.visibility_matrix[self.agent2.id], [True, True])

    def test_no_visibility_for_in_range_and_dead(self):
        self.agent2.is_alive = MagicMock(return_value=False) # declare dead

        self.world.occupy_pos(self.agent)
        self.world.occupy_pos(self.agent2)

        self.world._calculate_visibility(self.agent)
        self.world._calculate_visibility(self.agent2)

        np.testing.assert_array_equal(self.world.visibility_matrix[self.agent.id], [True, False])
        np.testing.assert_array_equal(self.world.visibility_matrix[self.agent2.id], [False, False])

    def test_visibility_for_not_in_range_and_alive(self):
        self.agent2.state.pos = np.array([3, 3]) # move out of range

        self.world.occupy_pos(self.agent)
        self.world.occupy_pos(self.agent2)

        self.world._calculate_visibility(self.agent)
        self.world._calculate_visibility(self.agent2)

        np.testing.assert_array_equal(self.world.visibility_matrix[self.agent.id], [True, False])
        np.testing.assert_array_equal(self.world.visibility_matrix[self.agent2.id], [False, True])


if __name__ == '__main__':
    unittest.main()
