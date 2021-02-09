import random

import numpy as np
from multiagent.core import World, Agent, WorldObject, Team
from multiagent.exceptions.scenario_exceptions import SymmetricScenarioSetupError
from multiagent.interfaces.scenario import BaseTeamScenario
from multiagent.reward_functions.dense_functions import reward_team_health, reward_team_damage
from multiagent.utils.colors import generate_colors
from multiagent.utils.spawn_generator import generate_spawns


class SymmetricTeamsScenario(BaseTeamScenario):
    def __init__(self, teams):
        """
        Constructor for a simple symmetric team scenario.
        @param teams: Team setup supplied as needed
        n_agents: How many agents per team
        n_teams: How many teams
        """
        self.team_build_plan = teams
        self.n_agents = [len(team) for team in teams]
        self.n_teams = len(teams)
        self.is_symmetric = False
        if self.is_symmetric and sum(self.n_agents) % self.n_teams != 0:
            raise SymmetricScenarioSetupError(self.n_agents, self.n_teams)

    def _make_world(self, grid_size):
        world = World(grid_size=grid_size)
        world.collaborative = True
        colors = generate_colors(self.n_teams)
        agent_count = 0
        for tid in range(self.n_teams):
            members = [
                Agent(
                    id=aid,  # This identifier is not reset per team. It is identifying all units unrelated to teams
                    name='Agent %d' % aid,
                    tid=tid,
                    color=colors[tid],
                    capabilities=self.team_build_plan[tid][index]['roles']
                ) for index, aid in
                enumerate(range(agent_count, agent_count + self.n_agents[tid]))
            ]
            agent_count += self.n_agents[tid]
            world.agents += members
            team = Team(tid=tid, members=members)
            world.teams.append(team)

        return world

    def reset_world(self, world):
        # random team spawns
        team_spawns = generate_spawns(*world.grid_center, self.n_teams, mean_radius=200)
        # scatter agents of a team a little
        for team, team_spawn in zip(world.teams, team_spawns):
            spawns = generate_spawns(*team_spawn, self.n_agents[team.tid], mean_radius=world.grid_size)
            for i, agent in enumerate(team.members):
                spawn = np.array(spawns[i])
                agent.state.reset(spawn)

    def reward(self, agent, world):
        reward = 0
        reward += reward_team_health(world.get_team(agent.tid))
        reward += reward_team_damage(world.get_team(agent.tid))
        return reward

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
