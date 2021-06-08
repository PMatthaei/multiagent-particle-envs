import unittest

from maenv.environment import MAEnv
from test.mock import mock_world, mock_agent, mock_team


class EnvironmentSpacesTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.a = mock_agent(id=0)
        self.b = mock_agent(id=1)
        self.c = mock_agent(id=2)
        self.d = mock_agent(id=3)
        self.agents_n = 4
        self.state_per_agent_dim = 6
        self.movement_dim = 4
        self.noop = 1
        t1 = mock_team(tid=0, members=[self.a, self.b], is_scripted=False)
        t2 = mock_team(tid=1, members=[self.c, self.d], is_scripted=True)
        self.world = mock_world(agents_n=self.agents_n, teams=[t1, t2])
        self.env = MAEnv(self.world, headless=True, observation_callback=lambda x, y: [])
        self.env.reset()

    def test_get_action_dim(self):
        dim = self.env._get_action_dim(self.a)
        # Each agent can perform an action against all other agents. he can move in 4 directions and do nothing
        self.assertEqual(dim, self.agents_n + self.movement_dim + self.noop)

    def test_get_state_dim(self):
        dim = self.env._get_state_dim()
        # State includes each agents data holding 6 features
        self.assertEqual(dim, self.state_per_agent_dim * self.agents_n)
