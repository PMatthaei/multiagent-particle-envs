from copy import deepcopy

from multiagent.core import RoleTypes, UnitAttackTypes
SMALL_1x1 = [
    {
        "is_scripted": False,
        "units": [  # Team 1
                     {
                         "role": RoleTypes.TANK,
                         "attack_type": UnitAttackTypes.RANGED
                     },
                 ] * 1
    },
    {
        "is_scripted": False,
        "units": [  # Team 2
                     {
                         "role": RoleTypes.TANK,
                         "attack_type": UnitAttackTypes.RANGED
                     },
                 ] * 1
    },
]
AI_SMALL_1x1 = deepcopy(SMALL_1x1)
AI_SMALL_1x1[0]["is_scripted"] = True

SMALL = [
    {
        "is_scripted": False,
        "units": [  # Team 1
                     {
                         "role": RoleTypes.TANK,
                         "attack_type": UnitAttackTypes.RANGED
                     },
                 ] * 5
    },
    {
        "is_scripted": False,
        "units": [  # Team 2
                     {
                         "role": RoleTypes.TANK,
                         "attack_type": UnitAttackTypes.RANGED
                     },
                 ] * 5
    },
]

AI_SMALL = deepcopy(SMALL)
AI_SMALL[0]["is_scripted"] = True

MEDIUM = [
    {
        "is_scripted": False,
        "units": [  # Team 1
                     {
                         "role": RoleTypes.TANK,
                         "attack_type": UnitAttackTypes.RANGED
                     },
                 ] * 25
    },
    {
        "is_scripted": False,
        "units": [  # Team 2
                     {
                         "role": RoleTypes.TANK,
                         "attack_type": UnitAttackTypes.RANGED
                     },
                 ] * 25
    },
]

AI_MEDIUM = deepcopy(MEDIUM)
AI_MEDIUM[0]["is_scripted"] = True

LARGE = [
    {
        "is_scripted": False,
        "units": [  # Team 1
                     {
                         "role": RoleTypes.TANK,
                         "attack_type": UnitAttackTypes.RANGED
                     },
                 ] * 100
    },
    {
        "is_scripted": False,
        "units": [  # Team 2
                     {
                         "role": RoleTypes.TANK,
                         "attack_type": UnitAttackTypes.RANGED
                     },
                 ] * 100
    },
]

AI_LARGE = deepcopy(LARGE)
AI_LARGE[0]["is_scripted"] = True

TWO_TEAMS_SIZE_TWO_ASYMMETRIC_HETEROGENEOUS = [
    {
        "is_scripted": True,
        "units": [  # Team 1
            {
                "role": RoleTypes.HEALER,
                "attack_type": UnitAttackTypes.RANGED
            },
            {
                "role": RoleTypes.TANK,
                "attack_type": UnitAttackTypes.RANGED
            },
        ]
    },
    {
        "is_scripted": False,
        "units": [  # Team 2
            {
                "role": RoleTypes.TANK,
                "attack_type": UnitAttackTypes.RANGED
            },
        ]
    },
]
TWO_TEAMS_SIZE_TWO_SYMMETRIC_HOMOGENEOUS_ADC = [
    {
        "is_scripted": False,
        "units": [  # Team 1
            {
                "role": RoleTypes.ADC,
                "attack_type": UnitAttackTypes.RANGED
            },
            {
                "role": RoleTypes.ADC,
                "attack_type": UnitAttackTypes.RANGED
            },
        ],
    },
    {
        "is_scripted": False,
        "units": [  # Team 2
            {
                "role": RoleTypes.ADC,
                "attack_type": UnitAttackTypes.RANGED
            },
            {
                "role": RoleTypes.ADC,
                "attack_type": UnitAttackTypes.RANGED
            },
        ],
    },
]

TWO_TEAMS_SIZE_TWO_SYMMETRIC_HOMOGENEOUS = [
    {
        "is_scripted": False,
        "units": [  # Team 1
            {
                "role": RoleTypes.TANK,
                "attack_type": UnitAttackTypes.RANGED
            },
            {
                "role": RoleTypes.TANK,
                "attack_type": UnitAttackTypes.RANGED
            },
        ],
    },
    {
        "is_scripted": False,
        "units": [  # Team 2
            {
                "role": RoleTypes.TANK,
                "attack_type": UnitAttackTypes.RANGED
            },
            {
                "role": RoleTypes.TANK,
                "attack_type": UnitAttackTypes.RANGED
            },
        ],
    },
]

TWO_TEAMS_SIZE_TWO_SYMMETRIC_HETEROGENEOUS = [
    {
        "is_scripted": False,
        "units": [  # Team 1
            {
                "role": RoleTypes.ADC,
                "attack_type": UnitAttackTypes.MELEE
            },
            {
                "role": RoleTypes.HEALER,
                "attack_type": UnitAttackTypes.RANGED
            },
        ],
    },
    {
        "is_scripted": False,
        "units": [  # Team 2
            {
                "role": RoleTypes.ADC,
                "attack_type": UnitAttackTypes.MELEE
            },
            {
                "role": RoleTypes.HEALER,
                "attack_type": UnitAttackTypes.RANGED
            },
        ],
    },
]

THREE_TEAMS_ASYMMETRIC_HETEROGENEOUS = [
    {
        "is_scripted": False,
        "units": [  # Team 1
            {
                "role": RoleTypes.ADC,
                "attack_type": UnitAttackTypes.MELEE
            },
            {
                "role": RoleTypes.TANK,
                "attack_type": UnitAttackTypes.RANGED
            },
            {
                "role": RoleTypes.TANK,
                "attack_type": UnitAttackTypes.RANGED
            }
        ],
    },
    {
        "is_scripted": False,
        "units": [  # Team 2
            {
                "role": RoleTypes.ADC,
                "attack_type": UnitAttackTypes.MELEE
            },
            {
                "role": RoleTypes.ADC,
                "attack_type": UnitAttackTypes.RANGED
            }
        ],
    },
    {
        "is_scripted": False,
        "units": [  # Team 3
            {
                "role": RoleTypes.HEALER,
                "attack_type": UnitAttackTypes.MELEE
            },
        ],
    },

]
