from maenv.core import Agent, World, Action


class ScriptedAI(object):

    def act(self, agent: Agent, world: World) -> Action:
        raise NotImplementedError()
