import unittest
from unittest.mock import MagicMock

import numpy as np

from maenv.core import World
from test.mock import mock_agent

N_AGENTS = 2


class WorldVisibilityTestCases(unittest.TestCase):
    def setUp(self):
        self.a = mock_agent(id=0)
        self.b = mock_agent(id=1, tid=1, sight_range=3)
        self.a_spawn = np.array([1, 1])
        self.b_spawn = np.array([1, 0])

        self.world = World(grid_size=10, teams_n=2, agents_n=N_AGENTS)
        self.world.agents = [self.a, self.b]

        self.world.connect(self.a, self.a_spawn)
        self.world.connect(self.b, self.b_spawn)

    def test_can_attack(self):
        self.world._update_visibility()

        result = self.world.can_attack(self.a, self.b)
        self.assertEqual(True, result)

    def test_visibility_for_in_range_and_alive(self):
        self.world._update_visibility()

        np.testing.assert_array_equal(self.world.visibility[self.a.id], [True, True])
        np.testing.assert_array_equal(self.world.visibility[self.b.id], [True, True])

        result = self.world.can_attack(self.a, self.b)
        self.assertEqual(True, result)

    def test_no_visibility_for_in_range_and_dead(self):
        self.b.is_alive = MagicMock(return_value=False)  # declare dead

        self.world.connect(self.b, self.b_spawn)

        self.world._update_visibility()

        np.testing.assert_array_equal(self.world.visibility[self.a.id], [True, False])
        np.testing.assert_array_equal(self.world.visibility[self.b.id], [False, False])

        result = self.world.can_attack(self.a, self.b)
        self.assertEqual(False, result)

    def test_no_visibility_for_not_in_range_and_alive(self):
        self.world.connect(self.b, np.array([30, 30]))  # move out of range

        self.world._update_visibility()

        np.testing.assert_array_equal(self.world.visibility[self.a.id], [True, False])
        np.testing.assert_array_equal(self.world.visibility[self.b.id], [False, True])

        result = self.world.can_attack(self.a, self.b)
        self.assertEqual(False, result)

    def test_no_visibility_for_not_in_range_and_dead(self):
        self.b.is_alive = MagicMock(return_value=False)  # declare dead
        self.world.connect(self.b, np.array([30, 30]))  # move out of range

        self.world._update_visibility()

        np.testing.assert_array_equal(self.world.visibility[self.a.id], [True, False])
        np.testing.assert_array_equal(self.world.visibility[self.b.id], [False, False])

        result = self.world.can_attack(self.a, self.b)
        self.assertEqual(False, result)


if __name__ == '__main__':
    unittest.main()
