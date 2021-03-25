import unittest

import numpy as np

from multiagent.core import Agent, RoleTypes, UnitAttackTypes, World
from multiagent.exceptions.agent_exceptions import OverhealError
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
        self.c = Agent(id=3, tid=0, build_plan=BUILD_PLAN_AGENT, color=None)

        self.a.state._health = [self.a.state.max_health]
        self.h.state._health = [self.h.state.max_health]
        self.b.state._health = [self.b.state.max_health]
        self.c.state._health = [self.c.state.max_health]

    def test_build_plan(self):
        self.assertEqual(self.a.unit_type_bits, [0, 0, 1])
        self.assertEqual(self.a.unit_type_bits_n, 3)
        self.assertEqual(self.a.attack_type, UnitAttackTypes.RANGED.value)
        self.assertEqual(self.a.role, RoleTypes.TANK.value)
        self.assertEqual(self.a.attack_range, UnitAttackTypes.RANGED.value['attack_range'])
        self.assertEqual(self.a.attack_range, self.a.sight_range)
        self.assertEqual(self.a.attack_damage, RoleTypes.TANK.value['attack_damage'])

    def test_attack(self):
        self.a.attack(self.b)
        self.assertEqual(self.b.state.max_health - self.a.attack_damage, self.b.state.health)

    def test_cannot_heal_if_not_healer(self):
        result = self.a.can_heal(self.c)
        self.assertEqual(False, result)

    def test_can_heal_if_alive_and_target_is_teammate_and_healer(self):
        self.c.state.health = self.c.state.max_health - 1
        result = self.h.can_heal(self.c)
        self.assertEqual(True, result)

    def test_cannot_heal_if_not_alive(self):
        result = self.a.can_heal(self.c)
        self.assertEqual(False, result)

    def test_cannot_heal_if_not_teammate(self):
        result = self.a.can_heal(self.b)
        self.assertEqual(False, result)

    def test_cannot_overheal_if_full_live(self):
        self.c.state.health = self.c.state.max_health
        result = self.h.can_heal(self.c)
        self.assertEqual(False, result)

    def test_overheal_raises_error(self):
        self.assertRaises(OverhealError, self.a.heal, self.c)
