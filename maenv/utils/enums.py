import json

from bin.team_plans_example import AI_SMALL
from maenv.core import RoleTypes, UnitAttackTypes

PUBLIC_ENUMS = {
    'RoleTypes': RoleTypes,
    'UnitAttackTypes': UnitAttackTypes,
}


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(PUBLIC_ENUMS[name], member)
    else:
        return d


if __name__ == '__main__':
    t = json.dumps(AI_SMALL, cls=EnumEncoder)
    print("JSON: ")
    print(t)
    t = json.loads(t, object_hook=as_enum)
    assert t == AI_SMALL
