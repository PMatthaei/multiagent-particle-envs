import unittest

import numpy as np

from maenv.ai.basic_ai import BasicScriptedAI
from test.mock import mock_agent, mock_team, mock_world

AGENTS_N = 4


class BasicAgentActTestCases(unittest.TestCase):
    def setUp(self):
        self.a = mock_agent(id=0, tid=0)
        self.b = mock_agent(id=1, tid=0)
        self.c = mock_agent(id=2, tid=1)
        self.d = mock_agent(id=3, tid=1)
        self.at = mock_team(0, members=[self.a, self.b])
        self.bt = mock_team(1, members=[self.c, self.d])
        self.world = mock_world(AGENTS_N, teams=[self.at, self.bt])
        self.world.positions = np.array([[0, 0], [0, 10], [0, 20], [0, 30]], dtype=float)
        self.world.distances = np.array([[0, 30, 20, 10], [30, 0, 20, 10], [30, 20, 0, 10], [10, 10, 10, 0]], dtype=float)
        self.ai = BasicScriptedAI()

    def test_a_should_attack_d(self):
        result = self.ai.act(self.a, self.world)
        np.testing.assert_array_equal(result.u, [0, 0, self.d.id])

    def test_b_should_attack_d(self):
        result = self.ai.act(self.b, self.world)
        np.testing.assert_array_equal(result.u, [0, 0, self.d.id])

    def test_c_should_not_attack(self):
        result = self.ai.act(self.c, self.world)
        np.testing.assert_array_equal(result.u, [0, -10, -1])

    def test_d_should_attack_a(self):
        result = self.ai.act(self.d, self.world)
        np.testing.assert_array_equal(result.u, [0, 0, self.a.id])

    def test_d_should_move_down_towards_b(self):
        # A and B are out of range -> no direct targeting -> -1
        # But B is closer (30 < 40) -> move to B
        self.world.distances = np.array([[0, 30, 20, 10], [30, 0, 20, 10], [20, 20, 0, 10], [40, 30, 10, 0]], dtype=float)
        result = self.ai.act(self.d, self.world)
        # Should move down to reach B -> B - D = (0,10) - (0,30) = (0,-20) -> move down by -10 (grid step)
        np.testing.assert_array_equal(result.u, [0, -10, -1])

    def test_a_should_move_up_towards_c(self):
        # C and D are out of range -> no direct targeting -> -1
        # But C is closer (30 < 40) -> move to C
        self.world.distances = np.array([[0, 10, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
        result = self.ai.act(self.a, self.world)
        # Should move up to reach C -> C - A = (0, 20) - (0, 0) = (0,20) -> move up by 10 (grid step)
        np.testing.assert_array_equal(result.u, [0, 10, -1])

    def test_a_should_move_right_towards_c(self):
        # C and D are out of range -> no direct targeting -> -1
        # But B is closer (3 < 4) -> move to B
        self.world.distances = np.array([[0, 10, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
        self.world.positions = np.array([[0, 0], [0, 0], [20, 0], [0, 0]], dtype=float)
        result = self.ai.act(self.a, self.world)
        # Should move up to reach C -> D - A = (20, 0) - (0, 0) = (20,0) -> move left by 10 (grid step)
        np.testing.assert_array_equal(result.u, [10, 0, -1])

    def test_a_should_move_left_towards_d(self):
        # C and D are out of range -> no direct targeting -> -1
        # But B is closer (30 < 40) -> move to B
        self.world.distances = np.array([[0, 10, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
        self.world.positions = np.array([[0, 0], [0, 00], [-20, 0], [0, 0]], dtype=float)
        result = self.ai.act(self.a, self.world)
        # Should move up to reach C -> D - A = (-20, 0) - (0, 0) = (-20,0) -> move left by 10 (grid step)
        np.testing.assert_array_equal(result.u, [-10, 0, -1])