import unittest

import numpy as np

from multiagent.environment import MAEnv
from test.mock import mock_world, mock_agent


class EnvironmentSetActionTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.a = mock_agent(id=0)
        self.world = mock_world(agents_n=2)
        self.env = MAEnv(self.world, headless=True)

    def test_sets_west_action(self):
        self.env._set_action(1, self.a)
        self.assertIsNotNone(self.a.action)
        np.testing.assert_array_equal(self.a.action.u, [-self.world.grid_size, 0, -1])

    def test_sets_east_action(self):
        self.env._set_action(2, self.a)
        self.assertIsNotNone(self.a.action)
        np.testing.assert_array_equal(self.a.action.u, [self.world.grid_size, 0, -1])

    def test_sets_north_action(self):
        self.env._set_action(3, self.a)
        self.assertIsNotNone(self.a.action)
        np.testing.assert_array_equal(self.a.action.u, [0, self.world.grid_size, -1])

    def test_sets_south_action(self):
        self.env._set_action(4, self.a)
        self.assertIsNotNone(self.a.action)
        np.testing.assert_array_equal(self.a.action.u, [0, -self.world.grid_size, -1])

    def test_sets_target_action_for_agent_with_id_0(self):
        self.env._set_action(5, self.a)
        self.assertIsNotNone(self.a.action)
        np.testing.assert_array_equal(self.a.action.u, [0, 0, 0])

    def test_sets_target_action_for_agent_with_id_3(self):
        self.env._set_action(8, self.a)
        self.assertIsNotNone(self.a.action)
        np.testing.assert_array_equal(self.a.action.u, [0, 0, 3])
