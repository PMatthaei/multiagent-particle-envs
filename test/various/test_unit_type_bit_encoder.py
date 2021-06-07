import unittest

import numpy as np

from utils import _unique_unit_types, bits_needed, unit_type_bits


class UnitTypeBitEncoderTestCases(unittest.TestCase):
    def setUp(self):
        self.types = [["Role_A", "Role_B"], ["Type_A", "Type_B"]]

    def test_unique_unit_two_types_two_roles(self):
        types = _unique_unit_types(types=self.types)
        self.assertEqual(len(types), 5)
        self.assertEqual(types, ['UNIT_TYPE_NONE',
                                 ('Role_A', 'Type_A'),
                                 ('Role_A', 'Type_B'),
                                 ('Role_B', 'Type_A'),
                                 ('Role_B', 'Type_B')])

    def test_unique_unit_one_types_two_roles(self):
        types = _unique_unit_types(types=[["Role_A", "Role_B"], ["Type_A"]])
        self.assertEqual(len(types), 3)
        self.assertEqual(types, ['UNIT_TYPE_NONE',
                                 ('Role_A', 'Type_A'),
                                 ('Role_B', 'Type_A')])

    def test_unique_unit_two_types_one_roles(self):
        types = _unique_unit_types(types=[["Role_A"], ["Type_A", "Type_B"]])
        self.assertEqual(len(types), 3)
        self.assertEqual(types, ['UNIT_TYPE_NONE',
                                 ('Role_A', 'Type_A'),
                                 ('Role_A', 'Type_B')])

    def test_bits_needed_for_one_type_one_role(self):
        n = bits_needed(types=[["Role_A"], ["Type_A"]])
        self.assertEqual(n, 1)  # 0 = Unknown and 1 = (Role_A, Type_A) -> One Bit suffices

    def test_bits_needed_for_two_type_one_role(self):
        n = bits_needed(types=[["Role_A"], ["Type_A", "Type_B"]])
        self.assertEqual(n, 2)  # 0 = Unknown and 1 = (Role_A, Type_A) -> One Bit suffices

    def test_unit_type_bits(self):
        unit_types = unit_type_bits(types=[["Role_A"], ["Type_A", "Type_B"]])
        np.testing.assert_array_equal(unit_types['UNIT_TYPE_NONE'], [0., 0.])
        np.testing.assert_array_equal(unit_types[('Role_A', 'Type_A')], [0., 1.])
        np.testing.assert_array_equal(unit_types[('Role_A', 'Type_B')], [1., 0.])
