import random

import numpy as np
from multiagent.core import World, Agent, WorldObject, Team
from multiagent.interfaces.scenario import BaseTeamScenario
from multiagent.utils.colors import generate_colors
from multiagent.utils.spawn_generator import generate_spawns


class SymmetricTeamsScenario(BaseTeamScenario):
    def __init__(self, n_agents=3, n_teams=2):
        """
        Constructor for a simple symmetric team scenario.
        @param n_agents: How many agents per team
        @param n_teams: How many teams
        """
        self.n_agents = n_agents
        self.n_teams = n_teams

    def _make_world(self, grid_size):
        world = World(grid_size=grid_size)
        world.collaborative = True
        colors = generate_colors(self.n_teams)
        for tid in range(self.n_teams):
            members = [Agent(id=aid, name='Agent %d' % aid, tid=tid, color=colors[tid]) for aid in
                       range(tid * self.n_agents, (tid + 1) * self.n_agents)]
            world.agents += members
            team = Team(tid=tid, members=members)
            world.teams.append(team)

        return world

    def reset_world(self, world):
        # random team spawns
        team_spawns = generate_spawns(world.grid_center[0], world.grid_center[1], self.n_teams, mean_radius=200)
        # scatter agents of a team a little
        for team, team_spawn in zip(world.teams, team_spawns):
            spawns = generate_spawns(team_spawn[0], team_spawn[1], self.n_agents, mean_radius=world.grid_size)
            for i, agent in enumerate(team.members):
                spawn = np.array(spawns[i])
                agent.state.reset(spawn)

    def reward(self, agent, world):
        return 1

    def done(self, agent, world):
        # if only one team left
        return [team.is_wiped() for team in world.teams].count(False) == 1

    def observation(self, agent, world):
        # Movement obs
        obs = [world.get_available_movement(agent)]
        # Ally obs
        for member in world.get_team_members(agent):
            obs.append(agent.observe(member))
        # Enemy obs
        for enemy in np.concatenate([team.members for team in world.get_opposing_teams(agent.tid)]):
            obs.append(agent.observe(enemy))
        # Self obs
        obs.append(agent.self_observation)
        return np.concatenate(obs)
