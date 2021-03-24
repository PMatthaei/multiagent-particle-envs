import unittest
from array import array

import numpy as np

from bin.team_plans_example import SMALL_1x1, AI_SMALL_1x1
from multiagent.scenarios.team.teams import TeamsScenario
from test.mock import mock_world, mock_team, mock_spawn_generator, mock_agent

AGENTS_N = 2


class TeamsScenarioDoneTestCases(unittest.TestCase):
    def setUp(self):
        self.a = mock_team(tid=0, members=[])
        self.b = mock_team(tid=1, members=[])
        self.world = mock_world(AGENTS_N, teams=[self.a, self.b])
        self.world.wiped_teams = [False, True]
        self.scenario = TeamsScenario(SMALL_1x1)

    def test_done_true_if_b_wiped_and_caller_a(self):
        result = self.scenario.done(self.a, self.world)
        self.assertTrue(result)

    def test_done_false_if_b_wiped_and_caller_b(self):
        result = self.scenario.done(self.b, self.world)
        self.assertFalse(result)

    def test_done_false_if_both_alive_and_caller_a_or_b(self):
        self.world.wiped_teams = [False, False]
        result = self.scenario.done(self.a, self.world)
        self.assertFalse(result)
        result = self.scenario.done(self.b, self.world)
        self.assertFalse(result)

    def test_done_true_if_both_wiped_and_caller_a_or_b(self):
        self.world.wiped_teams = [True, True]
        result = self.scenario.done(self.a, self.world)
        self.assertTrue(result)
        result = self.scenario.done(self.b, self.world)
        self.assertTrue(result)


class TeamsScenarioMakeTestCases(unittest.TestCase):
    def setUp(self):
        self.scenario = TeamsScenario(SMALL_1x1)
        self.ai_scenario = TeamsScenario(AI_SMALL_1x1)

    def test_make_world_creates_agents(self):
        result = self.scenario._make_world(grid_size=10)
        self.assertEqual(len(result.agents), 2)
        self.assertEqual(len(result.teams), 2)

    def test_make_world_creates_teams(self):
        result = self.scenario._make_world(grid_size=10)
        self.assertEqual(len(result.teams), 2)
        self.assertEqual(len(result.teams[0].members), 1)
        self.assertEqual(result.teams[0].tid, 0)
        self.assertEqual(len(result.teams[1].members), 1)
        self.assertEqual(result.teams[1].tid, 1)

    def test_make_world_creates_ai_team(self):
        result = self.ai_scenario._make_world(grid_size=10)
        self.assertEqual(len(result.teams), 2)
        self.assertEqual(len(result.teams[0].members), 1)
        self.assertEqual(result.teams[0].tid, 0)
        self.assertEqual(result.teams[0].is_scripted, True)


class TeamsScenarioResetTestCases(unittest.TestCase):
    def setUp(self):
        self.scenario = TeamsScenario(SMALL_1x1)
        self.c = mock_agent(id=0, tid=0)
        self.d = mock_agent(id=1, tid=1)
        self.a = mock_team(0, members=[self.c])
        self.b = mock_team(1, members=[self.d])
        self.world = mock_world(2, teams=[self.a, self.b])
        self.world.spg = mock_spawn_generator(n_teams=len(self.world.teams), n_agents=1)  # 1 agent per team

    def test_no_initial_spawns(self):
        self.assertIsNone(self.scenario.team_spawns)
        self.assertEqual(self.scenario.agent_spawns, [None, None])

    def test_reset_world_creates_spawns(self):
        self.scenario.reset_world(self.world)
        self.assertEqual(len(self.scenario.team_spawns), 2)
        self.assertEqual(len(self.scenario.agent_spawns), 2)

    def test_reset_world_calls_connect_agents(self):
        self.scenario.reset_world(self.world)
        self.assertEqual(self.world.connect.call_count, 2)
        result_arg_0 = self.world.connect.call_args[0][0]
        result_arg_1 = self.world.connect.call_args[0][1]
        self.assertEqual(self.d, result_arg_0)
        np.testing.assert_array_equal(np.array([0, 0]), result_arg_1)
