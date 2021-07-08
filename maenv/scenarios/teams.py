import numpy as np
from maenv.core import World, Agent, Team
from maenv.exceptions.scenario_exceptions import ScenarioNotSymmetricError
from maenv.interfaces.scenario import BaseTeamScenario
from maenv.utils.colors import generate_colors


class TeamsScenario(BaseTeamScenario):
    def __init__(self,
                 match_build_plan: dict,
                 grid_size: int = 10,
                 ai="basic", ai_config=None,
                 random_spawns: bool = False,
                 stochastic_spawns: bool = False,
                 **kwargs):
        """
        Constructor for a team scenario.
        @param match_build_plan: Plan to setup the match and therefore team composition and possible AI`s.
        n_agents: How many agents per team
        n_teams: How many teams
        """
        self.match_build_plan = match_build_plan
        assert match_build_plan is not None, "Cannot build scenario from empty build plan."
        self.grid_size = grid_size
        self.random_spawns = random_spawns
        self.stochastic_spawns = stochastic_spawns
        self.ai = ai
        self.ai_config = ai_config
        self.teams_n = len(match_build_plan)
        self.agents_n = [len(team["units"]) for team in match_build_plan]
        self.is_symmetric = self.agents_n.count(self.agents_n[0]) == len(self.agents_n)
        self.team_mixing_factor = 8  # build_plan["tmf"] if "tmf" in build_plan["tmf"] else 5
        if not self.is_symmetric:
            raise ScenarioNotSymmetricError(self.agents_n, self.teams_n)

        self.team_spawns = None
        if "agent_spawns" in self.match_build_plan:
            self.agent_spawns = self.match_build_plan["agent_spawns"]
        else:
            self.agent_spawns = [None] * self.teams_n

    def _make_world(self):
        """
        A teams scenario creates a world with two equally sized teams with either a fixed spawn scheme or
        a random generated spawn scheme. Spawns can be regenerated every episode or kept constant.
        @param grid_size:
        @return:
        """
        total_n_agents = sum(self.agents_n)

        world = World(n_agents=total_n_agents, n_teams=self.teams_n, grid_size=self.grid_size, ai=self.ai,
                      ai_config=self.ai_config)

        colors = generate_colors(self.teams_n)
        agent_count = 0
        for tid in range(self.teams_n):
            is_scripted = self.match_build_plan[tid]["is_scripted"]
            members = [
                Agent(
                    id=aid,  # is not reset per team. aid identifying all units globally
                    tid=tid,
                    color=colors[tid],
                    build_plan=self.match_build_plan[tid]["units"][index],
                    is_scripted=is_scripted,
                ) for index, aid in  # index is the team internal identifier
                enumerate(range(agent_count, agent_count + self.agents_n[tid]))
            ]
            agent_count += self.agents_n[tid]
            world.agents += members
            team = Team(tid=tid, members=members, is_scripted=is_scripted)
            world.teams.append(team)

        return world

    def reset_world(self, world: World):

        # How far should team spawns and agents be spread
        agent_spread = world.grid_size * sum(self.agents_n) / self.team_mixing_factor
        team_spread = self.teams_n * agent_spread

        # random team spawns
        if self.stochastic_spawns:  # if spawns already exist do not generate
            self.team_spawns = world.spg.generate_team_spawns(randomize=self.random_spawns, radius=team_spread)

        if self.stochastic_spawns:
            # take first teams size since symmetric for spawn generation
            agent_spawns = world.spg.generate(randomize=self.random_spawns, mean_radius=1, sigma_radius=agent_spread)
            # mirror spawns
            self.agent_spawns[0] = agent_spawns + self.team_spawns[0]
            self.agent_spawns[1] = (- agent_spawns) + self.team_spawns[1]

        for team, team_spawn in zip(world.teams, self.team_spawns):
            for team_intern_id, agent in enumerate(team.members):
                spawn = self.agent_spawns[team.tid][team_intern_id]
                world.connect(agent, spawn)

    def reward(self, agent: Agent, world: World):
        reward = 0
        reward += agent.stats.dmg_dealt / agent.attack_damage * 0.5
        reward += agent.stats.kills * 5
        return reward

    def done(self, team: Team, world: World):
        if np.all(world.wiped_teams):  # if all teams are wiped simultaneously -> done
            return True
        # if only one team is not wiped and this team is the team under testing -> winner winner chicken dinner
        return not world.wiped_teams[team.tid] and world.wiped_teams.count(False) == 1

    def observation(self, agent: Agent, world: World):
        other_obs = world.obs[agent.id].flatten()
        return np.concatenate((other_obs, agent.self_observation))
