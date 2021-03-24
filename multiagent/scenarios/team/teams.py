import numpy as np

from multiagent.ai.basic_ai import BasicScriptedAI
from multiagent.core import World, Agent, Team, Action
from multiagent.exceptions.scenario_exceptions import ScenarioNotSymmetricError
from multiagent.interfaces.scenario import BaseTeamScenario
from multiagent.utils.colors import generate_colors
from multiagent.utils.spawn_generator import SpawnGenerator


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
        self.n_agents = [len(team["units"]) for team in build_plan]
        self.is_symmetric = self.n_agents.count(self.n_agents[0]) == len(self.n_agents)
        self.team_mixing_factor = 8  # build_plan["tmf"] if "tmf" in build_plan["tmf"] else 5
        self.scripted_ai = BasicScriptedAI()
        if not self.is_symmetric:
            raise ScenarioNotSymmetricError(self.n_agents, self.n_teams)

        self.spg = None
        self.team_spawns = None
        self.agent_spawns = [None] * self.n_teams

    def _make_world(self, grid_size: int):
        agents_n = sum(self.n_agents)

        world = World(agents_n=agents_n, grid_size=grid_size)
        self.spg = SpawnGenerator(world, num_pos=agents_n)

        world.collaborative = True

        colors = generate_colors(self.n_teams)
        agent_count = 0
        for tid in range(self.n_teams):
            is_scripted = self.team_build_plan[tid]["is_scripted"]
            members = [
                Agent(
                    id=aid,  # is not reset per team. aid identifying all units globally
                    tid=tid,
                    color=colors[tid],
                    build_plan=self.team_build_plan[tid]["units"][index],
                    action_callback=self.scripted_agent_callback if is_scripted else None
                ) for index, aid in  # index is the team internal identifier
                enumerate(range(agent_count, agent_count + self.n_agents[tid]))
            ]
            agent_count += self.n_agents[tid]
            world.agents += members
            team = Team(tid=tid, members=members, is_scripted=is_scripted)
            world.teams.append(team)

        return world

    def reset_world(self, world: World):

        # How far should team spawns and agents be spread
        agent_spread = world.grid_size * sum(self.n_agents) / self.team_mixing_factor
        team_spread = self.n_teams * agent_spread

        # random team spawns
        if self.team_spawns is None:
            self.team_spawns = self.spg.generate_team_spawns(radius=team_spread, grid_size=world.grid_size)
            agent_spawns = self.spg.generate(self.n_agents[0], world.grid_size, 1, agent_spread)
            self.agent_spawns[0] = agent_spawns + self.team_spawns[0]
            self.agent_spawns[1] = (- agent_spawns) + self.team_spawns[1]

        # scatter agents of a team a little
        for team, team_spawn in zip(world.teams, self.team_spawns):
            for team_intern_id, agent in enumerate(team.members):
                spawn = self.agent_spawns[team.tid][team_intern_id]
                world.connect(agent, spawn)

    def reward(self, agent: Agent, world: World):
        reward = 0
        # reward += agent.state.health / agent.state.max_health
        # reward -= agent.stats.dmg_received / agent.state.max_health
        reward += agent.stats.dmg_dealt / agent.attack_damage * 0.5
        reward += agent.stats.kills * 5
        return reward

    def done(self, team: Team, world: World):
        # if only one team is not wiped and this team is the team under testing -> winner winner chicken dinner
        return not world.wiped_teams[team.tid] and world.wiped_teams.count(False) == 1

    def observation(self, agent: Agent, world: World):
        obs = world.obs[agent.id].flatten()
        return obs

    def scripted_agent_callback(self, agent: Agent, world: World) -> Action:
        return self.scripted_ai.act(agent, world)
