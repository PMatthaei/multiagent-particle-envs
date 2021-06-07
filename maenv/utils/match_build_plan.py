from typing import List, Tuple


class MatchBuildPlan:
    def __init__(self):
        self.teams = []
        self.agents = []
        pass

    def get_agent_spawns(self) -> List[Tuple[float, float]]:
        return list(map(lambda a: a.pos, self.agents))
