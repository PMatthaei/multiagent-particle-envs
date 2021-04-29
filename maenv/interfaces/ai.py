from maenv.core import Agent, World, Action


class ScriptedAI(object):
    def __init__(self):
        pass

    def act(self, agent: Agent, world: World) -> Action:
        """
        Act based upon the world and the agent state.
        @param agent:
        @param world:
        @return:
        """
        raise NotImplementedError()