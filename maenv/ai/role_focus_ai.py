from typing import List

import numpy as np

from maenv.ai import BasicScriptedAI
from maenv.core import World, RoleTypes


class FocusScriptedAI(BasicScriptedAI):
    def __init__(self, focuses: List[RoleTypes] = RoleTypes):
        """
        BasicAI with special targeting on pre-selected roles as focus.
        @param focuses: List of Roles ordered by importance of focus
        """
        super().__init__()
        self.focuses = focuses

    def _get_target(self, masked_distances: np.array, world: World):
        """
        Get the closest agent id matching the ordered list of focuses. The agent moves towards the focused target if
        it is not in range!
        @param masked_distances:
        @param world:
        @return:
        """
        targets = masked_distances.argsort()  # sort agent ids by distance
        target_roles = [world.agents[target].role_type for target in targets]  # agent roles
        for focus in self.focuses:  # search for each focus until the closest possible target is found
            for target in targets:
                target_role = target_roles[target]
                if target_role == focus:
                    return target

        return targets[0]
