from multiagent.core import RoleTypes, UnitAttackTypes

TWO_TEAMS_SIZE_TWO_SYMMETRIC_HOMOGENEOUS = [
    [  # Team 1
        {
            "roles": [RoleTypes.TANK],
            "attack": [UnitAttackTypes.RANGED]
        },
        {
            "roles": [RoleTypes.TANK],
            "attack": [UnitAttackTypes.RANGED]
        },
    ],
    [  # Team 2
        {
            "roles": [RoleTypes.TANK],
            "attack": [UnitAttackTypes.RANGED]
        },
        {
            "roles": [RoleTypes.TANK],
            "attack": [UnitAttackTypes.RANGED]
        },
    ],
]

TWO_TEAMS_SIZE_TWO_SYMMETRIC_HETEROGENEOUS = [
    [  # Team 1
        {
            "roles": [RoleTypes.ADC],
            "attack": [UnitAttackTypes.MELEE]
        },
        {
            "roles": [RoleTypes.TANK],
            "attack": [UnitAttackTypes.RANGED]
        },
    ],
    [  # Team 2
        {
            "roles": [RoleTypes.ADC],
            "attack": [UnitAttackTypes.MELEE]
        },
        {
            "roles": [RoleTypes.TANK],
            "attack": [UnitAttackTypes.RANGED]
        },
    ],
]

THREE_TEAMS_ASYMMETRIC_HETEROGENEOUS = [
    [  # Team 1
        {
            "roles": [RoleTypes.ADC],
            "attack": [UnitAttackTypes.MELEE]
        },
        {
            "roles": [RoleTypes.TANK],
            "attack": [UnitAttackTypes.RANGED]
        },
        {
            "roles": [RoleTypes.TANK],
            "attack": [UnitAttackTypes.RANGED]
        }
    ],
    [  # Team 2
        {
            "roles": [RoleTypes.ADC],
            "attack": [UnitAttackTypes.MELEE]
        },
        {
            "roles": [RoleTypes.ADC],
            "attack": [UnitAttackTypes.RANGED]
        }
    ],
    [  # Team 3
        {
            "roles": [RoleTypes.HEALER],
            "attack": [UnitAttackTypes.MELEE]
        },
    ],
]