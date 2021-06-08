import unittest
import numpy as np
from maenv.environment import MAEnv
from test.mock import mock_agent, mock_team, mock_world


class EnvironmentStepTestCases(unittest.TestCase):

    def setUp(self) -> None:
        self.a = mock_agent(id=0)
        self.b = mock_agent(id=1)
        self.c = mock_agent(id=2, tid=1)
        self.d = mock_agent(id=3, tid=1)
        self.at = mock_team(tid=0, members=[self.a, self.b])
        self.bt = mock_team(tid=1, members=[self.c, self.d])
        self.world = mock_world(agents_n=2, teams=[self.at, self.bt])
        actions_n = len(self.at.members)
        self.dummy_action = np.ones((actions_n,))
        obs_n = 10
        obs = np.zeros((obs_n,))
        self.env = MAEnv(self.world, headless=True,
                         observation_callback=lambda agent, world: obs,  # all agents have same obs - obs mock
                         done_callback=lambda team, world: team.tid == 0,  # Team with ID 0 wins - done mock
                         reward_callback=lambda agent, world: 1.0)  # Local per agent reward mock

    def test_env_step_calls_world_step(self):
        self.env.step(self.dummy_action)
        self.assertEqual(self.world.step.call_count, 1)

    def test_env_step_increases_timestep_count(self):
        self.assertEqual(self.env.t, 0)
        self.env.step(self.dummy_action)
        self.assertEqual(self.env.t, 1)

    def test_env_step_increases_timestep_count_10times(self):
        self.assertEqual(self.env.t, 0)
        for _ in range(10):
            self.env.step(self.dummy_action)
        self.assertEqual(self.env.t, 10)

    def test_env_step_return_obs_for_all_policy_agents(self):
        obs_n, reward_n, done_n, info_n = self.env.step(self.dummy_action)
        self.assertEqual(len(obs_n), len(self.at.members))

    def test_env_step_returns_global_reward_for_policy_team(self):
        obs_n, reward_n, done_n, info_n = self.env.step(self.dummy_action)
        self.assertEqual(len(reward_n), 1)  # reward is a single value
        self.assertEqual(reward_n[0], 201)  # reward

    def test_env_step_returns_local_reward_for_policy_team(self):
        self.env.global_reward = False
        obs_n, reward_n, done_n, info_n = self.env.step(self.dummy_action)
        self.assertEqual(len(reward_n), 2)  # reward is a single value
        self.assertEqual(reward_n[0], 101)  # reward

    def test_env_step_returns_winning_team_a(self):
        obs_n, reward_n, done_n, info_n = self.env.step(self.dummy_action)
        self.assertTrue(done_n[self.at.tid])

    def test_env_step_returns_not_winning_team_b(self):
        obs_n, reward_n, done_n, info_n = self.env.step(self.dummy_action)
        self.assertFalse(done_n[self.bt.tid])

    def test_env_step_returns_no_draw(self):
        obs_n, reward_n, done_n, info_n = self.env.step(self.dummy_action)
        self.assertFalse(info_n["draw"])

    def test_env_step_returns_draw(self):
        self.env.done_callback = lambda team, world: True  # all teams win -> draw
        obs_n, reward_n, done_n, info_n = self.env.step(self.dummy_action)
        self.assertTrue(info_n["draw"])

    def test_env_step_returns_draw_if_step_limit_reached(self):
        self.env.done_callback = lambda team, world: False  # no one has won
        self.env.episode_limit = 1
        obs_n, reward_n, done_n, info_n = self.env.step(self.dummy_action)
        self.assertTrue(info_n["draw"])
