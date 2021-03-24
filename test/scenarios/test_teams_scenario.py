import unittest

from bin.team_plans_example import SMALL_1x1
from multiagent.scenarios.team.teams import TeamsScenario
from test.mock import mock_world, mock_team

AGENTS_N = 2


class TeamsScenarioTestCases(unittest.TestCase):
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