import numpy as np

from multiagent.core import World, Agent, Team, RoleTypes
from multiagent.exceptions.scenario_exceptions import ScenarioNotSymmetricError, SymmetricScenarioTeamsExceededError
from multiagent.interfaces.scenario import BaseTeamScenario
from multiagent.reward_functions.dense_functions import reward_team_health, reward_team_damage
from multiagent.utils.colors import generate_colors
from multiagent.utils.spawn_generator import generate_spawns


class TeamsScenario(BaseTeamScenario):
    def __init__(self, build_plan):
        """
        Constructor for a team scenario.
        @param build_plan: Team setup supplied as needed
        n_agents: How many agents per team
        n_teams: How many teams
        """
        self.team_build_plan = build_plan
        self.n_teams = len(build_plan)
        self.n_agents = [len(team) for team in build_plan]
        self.is_symmetric = build_plan.count(build_plan[0]) == len(build_plan)

        if self.is_symmetric and sum(self.n_agents) % self.n_teams != 0:
            raise ScenarioNotSymmetricError(self.n_agents, self.n_teams)

    def _make_world(self, grid_size):
        world = World(grid_size=grid_size)
        world.collaborative = True
        colors = generate_colors(self.n_teams)
        agent_count = 0
        for tid in range(self.n_teams):
            members = [
                Agent(
                    id=aid,  # is not reset per team. aid identifying all units globally
                    tid=tid,
                    color=colors[tid],
                    build_plan=self.team_build_plan[tid][index]
                ) for index, aid in  # index is the team internal identifier
                enumerate(range(agent_count, agent_count + self.n_agents[tid]))
            ]
            agent_count += self.n_agents[tid]
            world.agents += members
            team = Team(tid=tid, members=members)
            world.teams.append(team)

        return world

    def reset_world(self, world):
        # random team spawns
        team_spawns = generate_spawns(*world.grid_center, self.n_teams, mean_radius=world.grid_size * 3)
        # scatter agents of a team a little
        for team, team_spawn in zip(world.teams, team_spawns):
            spawns = generate_spawns(*team_spawn, self.n_agents[team.tid], mean_radius=world.grid_size)
            for i, agent in enumerate(team.members):
                spawn = np.array(spawns[i])
                agent.state.reset(spawn)

    def reward(self, agent, world):
        reward = 0
        reward += agent.state.health / agent.state.max_health
        reward += agent.stats.dmg_dealt
        return reward

    def done(self, agent, world):
        # if only one team left
        return [team.is_wiped() for team in world.teams].count(False) == 1

    def observation(self, agent, world):
        # Movement observation of the agent
        obs = [world.get_available_movement(agent)]
        # Ally observation
        for member in world.get_team_members(agent):
            obs.append(agent.observe(member))
        # Enemy observation
        for enemy in np.concatenate([team.members for team in world.get_opposing_teams(agent.tid)]):
            obs.append(agent.observe(enemy))
        # Self observation
        obs.append(agent.self_observation)
        return np.concatenate(obs)
