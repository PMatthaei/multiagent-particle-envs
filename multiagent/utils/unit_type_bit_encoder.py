# Calculate all unique unit types
import itertools
import math
from typing import List

UNKNOWN_TYPE = "UNIT_TYPE_NONE"


def _unique_unit_types(types: List):
    """
    How many units can be formed with the types provided as a list.
    @param types:
    @return:
    """
    types = list(itertools.product(*types))
    types.insert(0, UNKNOWN_TYPE)  # Add type if unit not observable
    return types


def bits_needed(types: List) -> int:
    """
    How many bits are needed to represent the types provided in the list.
    @param types:
    @return:
    """
    return math.ceil(math.log(len(_unique_unit_types(types)), 2))


def _to_bits(index: int, types):
    """
    Convert a type with a given index in the unique type set into bit representation.
    @param index:
    @param types:
    @return:
    """
    return list(map(float, bin(index)[2:].zfill(bits_needed(types))))


def unit_type_bits(types: List) -> dict:
    """
    Convert all unique types into bit representation.
    @param types:
    @return:
    """
    return dict((unit, _to_bits(index, types)) for index, unit in enumerate(_unique_unit_types(types)))
