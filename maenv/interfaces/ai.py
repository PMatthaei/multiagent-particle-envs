import numpy as np

from maenv.core import Agent, World, Action


class ScriptedAI(object):

    def act(self, agent: Agent, world: World) -> Action:
        raise NotImplementedError()

    def _get_target(self, world) -> int:
        """
        Implement basic targeting with the provided masked distances.
        @param masked_distances:
        @param world:
        @return: id of the target
        """
        raise NotImplementedError()

    def _get_masked_distances(self, agent: Agent, world: World) -> np.array:
        """
        Implement how distances to other agents should be masked out in order to define which agents should be considered
        for targeting.
        @param agent:
        @param world:
        @return:
        """
        raise NotImplementedError()

