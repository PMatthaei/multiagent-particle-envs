import unittest

import numpy as np

from maenv.environment import MAEnv
from test.mock import mock_world, mock_agent, mock_team


class EnvironmentAvailableActionTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.a = mock_agent(id=0)
        self.b = mock_agent(id=1)
        self.c = mock_agent(id=2)
        self.d = mock_agent(id=3)
        team_1 = mock_team(tid=0, members=[self.a, self.b], is_scripted=False)
        team_2 = mock_team(tid=1, members=[self.c, self.d], is_scripted=True)
        self.world = mock_world(agents_n=2, teams=[team_1, team_2])
        self.env = MAEnv(self.world, headless=True)
        self.env.reset()

    def test_get_avail_actions_healer(self):
        avail_actions = self.env.get_avail_actions()
        np.testing.assert_array_equal(avail_actions, [1])
