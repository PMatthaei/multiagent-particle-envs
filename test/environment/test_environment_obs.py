import unittest

import numpy as np

from maenv.environment import MAEnv
from test.mock import mock_world, mock_agent, mock_team


class EnvironmentObservationsTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.a = mock_agent(id=0)
        self.b = mock_agent(id=1)
        self.c = mock_agent(id=2)
        self.d = mock_agent(id=3)

        self.obs_n = 12
        self.state_per_agent_dim = 6
        self.movement_dim = 4
        self.noop = 1

        t1 = mock_team(tid=0, members=[self.a, self.b], is_scripted=False)
        t2 = mock_team(tid=1, members=[self.c, self.d], is_scripted=True)
        self.world = mock_world(teams=[t1, t2])

        self.policy_agents = len(self.world.policy_agents)
        self.agents_n = len(self.world.agents)

        self.env = MAEnv(self.world, headless=True, observation_callback=lambda x, y: list(range(self.obs_n)))
        self.env.reset()

    def test_get_obs(self):
        obs = self.env.get_obs()
        obs = np.array(obs)
        self.assertEqual(obs.shape, (self.policy_agents, self.obs_n))