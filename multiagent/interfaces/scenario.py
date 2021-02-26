# defines scenario upon which the world is built
from multiagent.core import World, Team, Agent


class BaseScenario(object):
    def _make_world(self) -> World:
        """
        Create elements of the world
        :return: world
        """
        raise NotImplementedError()

    def reset_world(self, world: World):
        """
        Create initial conditions of the world
        :param world:
        :return: None
        """
        raise NotImplementedError()

    def reward(self, agent: Agent, world: World):
        """
        Reward an agents action with a individual/local reward based on world and agent data
        :param agent:
        :param world:
        :return:
        """
        raise NotImplementedError()

    def observation(self, agent: Agent, world: World):
        """
        Return observation of the agent in the given world
        :param agent:
        :param world:
        :return: obs
        """
        raise NotImplementedError()

    def done(self, agent: Agent, world: World):
        """
        Return if this agent has achieved his goal
        :param agent:
        :param world:
        :return:
        """
        raise NotImplementedError()


class BaseTeamScenario(object):
    def make_teams_world(self, grid_size=10):
        """
        This function needs to be called instead of make_world to create a team-based setup
        :return:
        """
        world = self._make_world(grid_size)
        # make initial conditions
        self.reset_world(world)
        return world

    # create elements of the world
    def _make_world(self, grid_size: int) -> World:
        """
        Create elements of the world
        :return: world
        """
        raise NotImplementedError()

    def reset_world(self, world: World):
        """
        Create initial conditions of the world
        :param world:
        :return: None
        """
        raise NotImplementedError()

    def reward(self, agent: Agent, world: World):
        """
        Reward an agents action individual/local based on world and agent data
        :param agent:
        :param world:
        :return:
        """
        raise NotImplementedError()

    def observation(self, agent: Agent, world: World):
        """
        Return observation of the agent in the given world
        :param agent:
        :param world:
        :return: obs
        """
        raise NotImplementedError()

    def done(self, team: Team, world: World):
        """
        Return if this team has achieved his goal
        :param team:
        :param world:
        :return:
        """
        raise NotImplementedError()
