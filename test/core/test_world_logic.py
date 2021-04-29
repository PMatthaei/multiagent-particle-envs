import unittest

import numpy as np

from maenv.core import World, Agent, RoleTypes, UnitAttackTypes
from test.mock import mock_team

N_AGENTS = 2

BUILD_PLAN = {
    "role": RoleTypes.TANK,
    "attack_type": UnitAttackTypes.RANGED
}

BUILD_PLAN_HEALER = {
    "role": RoleTypes.HEALER,
    "attack_type": UnitAttackTypes.RANGED
}


class WorldLogicTestCases(unittest.TestCase):
    def setUp(self):
        self.a = Agent(id=0, tid=0, build_plan=BUILD_PLAN, color=None)
        self.b = Agent(id=1, tid=1, build_plan=BUILD_PLAN, color=None)

        self.world = World(grid_size=10, teams_n=2, agents_n=2)
        self.world.agents = [self.a, self.b]
        team_a = mock_team(tid=0, members=[self.a])
        team_b = mock_team(tid=1, members=[self.b])
        self.world.teams = [team_a, team_b]

        self.world.connect(self.a)
        self.world.connect(self.b)

        self.world._calculate_wiped_teams()

    def test_update_alive_all_true(self):
        self.world._update_alive_status()
        np.testing.assert_array_equal(self.world.alive, [True, True])

    def test_update_alive_of_a_to_false(self):
        self.a.state.health = 0  # declare dead
        self.world._update_alive_status()
        np.testing.assert_array_equal(self.world.alive, [False, True])

    def test_update_alive_of_b_to_false(self):
        self.b.state.health = 0  # declare dead
        self.world._update_alive_status()
        np.testing.assert_array_equal(self.world.alive, [True, False])

    def test_teams_wiped_all_false(self):
        np.testing.assert_array_equal(self.world.wiped_teams, [False, False])

    def test_teams_wiped_team_a_true(self):
        self.a.state.health = 0  # declare dead -> team dead
        self.world._update_alive_status()
        self.world._calculate_wiped_teams()
        np.testing.assert_array_equal(self.world.wiped_teams, [True, False])

    def test_teams_wiped_team_b_true(self):
        self.b.state.health = 0  # declare dead -> team dead
        self.world._update_alive_status()
        self.world._calculate_wiped_teams()
        np.testing.assert_array_equal(self.world.wiped_teams, [False, True])

    def test_teams_wiped_all_true(self):
        self.a.state.health = 0  # declare dead -> team dead
        self.b.state.health = 0  # declare dead -> team dead
        self.world._update_alive_status()
        self.world._calculate_wiped_teams()
        np.testing.assert_array_equal(self.world.wiped_teams, [True, True])
