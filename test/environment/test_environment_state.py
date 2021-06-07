import unittest

import numpy as np

from maenv.environment import MAEnv
from test.mock import mock_world, mock_agent, mock_team


class EnvironmentStateTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.a = mock_agent(id=0)
        self.b = mock_agent(id=1, tid=1)
        self.at = mock_team(tid=0, members=[self.a])
        self.bt = mock_team(tid=1, members=[self.b])
        self.world = mock_world(agents_n=2, teams=[self.at, self.bt])
        self.env = MAEnv(self.world, headless=True, observation_callback=lambda x,y: [])

    def test_get_state_returns_both_in_top_left_corner_relative_to_world_center(self):
        result = self.env.get_state()
        np.testing.assert_array_equal(result, [-0.5, -0.5, 1., 0., 0., 1., -0.5, -0.5, 1., 0., 0., 1.])

    def test_get_state_returns_b_in_center_relative_to_world_center(self):
        self.b.state.pos = self.world.center # Place in B in map center
        result = self.env.get_state()
        np.testing.assert_array_equal(result, [-0.5, -0.5, 1., 0., 0., 1., 0., 0., 1., 0., 0., 1.])
