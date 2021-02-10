from multiagent.core import RoleTypes, UnitAttackTypes

TWO_TEAMS_SIZE_TWO_SYMMETRIC_HOMOGENEOUS = [
    [  # Team 1
        {
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
    ],
    [  # Team 2
        {
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
    ],
]

TWO_TEAMS_SIZE_TWO_SYMMETRIC_HETEROGENEOUS = [
    [  # Team 1
        {
            "role": RoleTypes.ADC,
            "attack_type": UnitAttackTypes.MELEE
        },
        {
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
    ],
    [  # Team 2
        {
            "role": RoleTypes.ADC,
            "attack_type": UnitAttackTypes.MELEE
        },
        {
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
    ],
]

THREE_TEAMS_ASYMMETRIC_HETEROGENEOUS = [
    [  # Team 1
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
    [  # Team 2
        {
            "role": RoleTypes.ADC,
            "attack_type": UnitAttackTypes.MELEE
        },
        {
            "role": RoleTypes.ADC,
            "attack_type": UnitAttackTypes.RANGED
        }
    ],
    [  # Team 3
        {
            "role": RoleTypes.HEALER,
            "attack_type": UnitAttackTypes.MELEE
        },
    ],
]