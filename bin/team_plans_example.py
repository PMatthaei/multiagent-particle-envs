from multiagent.core import RoleTypes, UnitAttackTypes

LARGE = [
    {
        "is_scripted": True,
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
