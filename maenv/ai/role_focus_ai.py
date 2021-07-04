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
        self.focuses = [int(f) for f in focuses]
        self.target_role_mask = None

    def _get_target(self, world: World) -> int:
        """
        Get the closest agent id matching the ordered list of focuses. The agent moves towards the focused target if
        it is not in range!
        @param masked_distances:
        @param world:
        @return: id of the target
        """
        if self.target_role_mask is None:  # init role mask on first run
            self.target_role_mask = np.array([int(agent.role_type) for agent in world.agents])
        focus_masked_distances = self.masked_distances.copy()
        for focus in self.focuses:  # search for each focus until the closest possible target is found
            no_focus_mask = self.target_role_mask != focus
            focus_masked_distances[no_focus_mask] = np.inf  # mask out agents not in focus because of their role
            no_focus_target = np.all(np.isinf(focus_masked_distances))
            if no_focus_target:
                continue # search in next focus
            else:
                return np.argmin(focus_masked_distances)

        return np.argmin(self.masked_distances)  # no target is of the desired focus role -> fallback: just attack closest
