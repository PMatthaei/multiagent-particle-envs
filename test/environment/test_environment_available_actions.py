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
        agents_n = 4
        t1 = mock_team(tid=0, members=[self.a, self.b], is_scripted=False)
        t2 = mock_team(tid=1, members=[self.c, self.d], is_scripted=True)
        self.world = mock_world(agents_n=agents_n, teams=[t1, t2])
        self.world.avail_target_actions = np.array([
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ])
        self.assertEqual(self.world.avail_target_actions.shape, (agents_n, agents_n))
        self.world.avail_movement_actions = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.assertEqual(self.world.avail_target_actions.shape, (4, 4))  # 4 movement dims in 2d space
        self.env = MAEnv(self.world, headless=True, observation_callback=lambda x, y: [])
        self.env.reset()

    def test_get_avail_actions_agent_a(self):
        avail_action_ids = self.env.get_available_action_ids(self.a)
        np.testing.assert_array_equal(avail_action_ids, [0, 1, 7, 8])
        avail_actions = self.env.get_available_actions(self.a)
        np.testing.assert_array_equal(avail_actions, [1., 1., 0., 0., 0., 0., 0., 1., 1.])

    def test_get_avail_actions_agent_b(self):
        avail_action_ids = self.env.get_available_action_ids(self.b)
        np.testing.assert_array_equal(avail_action_ids, [0, 2, 5, 6])
        avail_actions = self.env.get_available_actions(self.b)
        np.testing.assert_array_equal(avail_actions, [1., 0., 1., 0., 0., 1., 1., 0., 0.])

    def test_get_avail_actions_agent_c(self):
        avail_action_ids = self.env.get_available_action_ids(self.c)
        np.testing.assert_array_equal(avail_action_ids, [0, 3])
        avail_actions = self.env.get_available_actions(self.c)
        np.testing.assert_array_equal(avail_actions, [1., 0., 0., 1., 0., 0., 0., 0., 0.])

    def test_get_avail_actions_agent_d(self):
        avail_action_ids = self.env.get_available_action_ids(self.d)
        np.testing.assert_array_equal(avail_action_ids, [0, 4, 5])
        avail_actions = self.env.get_available_actions(self.d)
        np.testing.assert_array_equal(avail_actions, [1., 0., 0., 0., 1., 1., 0., 0., 0.])

    def test_get_avail_actions_for_all_policy_agents(self):
        avail_actions = self.env.get_avail_actions()
        avail_actions = np.array(avail_actions)
        self.assertEqual(avail_actions.shape, (2, 9))
        np.testing.assert_array_equal(avail_actions, [
            [1., 1., 0., 0., 0., 0., 0., 1., 1.],  # Agent A
            [1., 0., 1., 0., 0., 1., 1., 0., 0.]  # Agent B
        ])
