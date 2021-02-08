import random

import numpy as np
from multiagent.core import World, Agent, WorldObject, Team
from multiagent.interfaces.scenario import BaseTeamScenario


class TeamSimpleScenario(BaseTeamScenario):
    def _build_teams(self, world):
        length = len(world.agents)
        middle_index = length // 2

        team_1 = Team(0, world.agents[:middle_index])
        world.teams.append(team_1)

        team_2 = Team(1, world.agents[middle_index:])
        world.teams.append(team_2)

        for i, team in enumerate(world.teams):
            team_color = [0.75, 0, 0] if i == 0 else [0, 0, 0.75]
            for agent in team.members:
                agent.color = np.array(team_color)

    def _make_world(self, grid_size):
        # TODO: Create world by template f.e. json or other
        world = World(grid_size=grid_size)
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(4)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.silent = True
        # add landmarks
        world.objects = [WorldObject() for i in range(1)]
        for i, landmark in enumerate(world.objects):
            landmark.name = 'landmark %d' % i
            landmark.movable = False
        return world

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.objects):
            landmark.color = np.array([0, 0.75, 0])
        # set random initial states
        for i, team in enumerate(world.teams):
            for agent in team.members:
                side = -1 if i == 0 else +1  # team 0 spawns left
                spawn = np.array([world.grid_center[0] + 4 * world.grid_size * side, world.grid_center[1]])
                agent.state.reset(spawn)
        for i, landmark in enumerate(world.objects):
            landmark.state.pos = np.array([world.grid_center[0], world.grid_center[1]])

    def reward(self, agent, world):
        max_distance = np.linalg.norm(world.bounds - world.objects[0].state.pos)
        r = np.linalg.norm(agent.state.pos - world.objects[0].state.pos) / max_distance
        r = 1 - r
        if agent.can_see(world.objects[0]):
            r += 100
        return r

    def done(self, agent, world):
        return agent.can_see(world.objects[0])

    def observation(self, agent, world):
        obs = []
        for member in world.get_team_members(agent):
            obs.append(agent.observe(member))
        for enemy in np.concatenate([team.members for team in world.get_opposing_teams(agent.tid)]):
            obs.append(agent.observe(enemy))
        return np.concatenate(obs)
