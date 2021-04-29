import unittest

import numpy as np

from maenv.core import Agent, RoleTypes, UnitAttackTypes, World
from maenv.exceptions.agent_exceptions import OverhealError
from test.mock import mock_world

BUILD_PLAN_AGENT = {
    "role": RoleTypes.TANK,
    "attack_type": UnitAttackTypes.RANGED
}

BUILD_PLAN_HEALER = {
    "role": RoleTypes.HEALER,
    "attack_type": UnitAttackTypes.RANGED
}

N_AGENTS = 4


class AgentTestCases(unittest.TestCase):
    def setUp(self):
        self.a = Agent(id=0, tid=0, build_plan=BUILD_PLAN_AGENT, color=None)
        self.h = Agent(id=1, tid=0, build_plan=BUILD_PLAN_HEALER, color=None)
        self.b = Agent(id=2, tid=1, build_plan=BUILD_PLAN_AGENT, color=None)
        self.c = Agent(id=3, tid=1, build_plan=BUILD_PLAN_AGENT, color=None)

        self.a.state._health = [self.a.state.max_health]
        self.b.state._health = [self.b.state.max_health]
        self.c.state._health = [self.a.attack_damage]

    def test_attack_dmg_dealt_stats_received(self):
        self.a.attack(self.b)
        self.assertEqual(self.a.attack_damage, self.a.stats.dmg_dealt)

    def test_attack_kills_stats_received(self):
        self.a.attack(self.c)
        self.assertEqual(self.a.attack_damage, self.a.stats.dmg_dealt)
        self.assertEqual(1, self.a.stats.kills)

    def test_heal_amount_stats_received(self):
        self.a.state._health = [self.a.state.max_health - self.h.attack_damage]
        self.h.heal(self.a)
        self.assertEqual(self.h.attack_damage, self.h.stats.dmg_healed)
