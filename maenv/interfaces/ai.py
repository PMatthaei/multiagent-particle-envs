import numpy as np

from maenv.core import Agent, World, Action


class ScriptedAI(object):
    def __init__(self):
        self.action = Action()

    def act(self, agent: Agent, world: World) -> Action:
        """
        Act based upon the world and the agent state.
        @param agent:
        @param world:
        @return:
        """
        self.action.u = np.zeros(world.dim_p + 1)  # reset previous action
        self.action.u[2] = -1  # default is no target == -1
        return self.action