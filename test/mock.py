from functools import reduce
from unittest.mock import MagicMock, Mock

import numpy as np


def mock_agent(id: int, tid: int = 0, sight_range=2, pos=np.array([0, 0])):
    agent = Mock()
    agent.id = id
    agent.tid = tid
    agent.action = Mock()
    agent.state.max_health = 100
    agent.state._health = np.array([100])
    agent.state.health = MagicMock(return_value=agent.state._health[0])
    agent.is_alive = MagicMock(return_value=True)
    agent.sight_range = sight_range
    agent.has_heal = MagicMock(return_value=False)
    agent.action.u = np.zeros((2,))
    agent.unit_type_bits = [0, 0, 1]
    agent.state.pos = pos
    agent.self_observation = [agent.state.health() / agent.state.max_health] + agent.unit_type_bits

    return agent


def mock_team(tid: int, members=None, is_scripted=False):
    if members is None:
        members = []
    team = Mock(tid=tid, members=members, is_scripted=is_scripted)
    return team


def mock_world(agents_n, grid_size=10, teams=None, obs_dims_per_agent=8, teams_n=2):
    if teams is None:
        teams = []
    world = Mock(agents_n=agents_n, teams_n=teams_n, grid_size=grid_size)
    world.bounds = np.array([1280, 720])
    world.center = world.bounds / 2
    world.agents = reduce(lambda agents, team: agents + team.members, teams, [])
    world.team_affiliations = np.array(list(map(lambda a: a.tid, world.agents)))
    world.teams = teams
    world.policy_agents = []
    world.dim_p = 2
    world.connect = MagicMock()
    world.obs = np.zeros((agents_n, agents_n, int(obs_dims_per_agent * agents_n / 2)))
    world.obs[0, :] = 1.0
    return world


def mock_spawn_generator(n_teams, n_agents):
    spg = Mock()
    spg.generate_team_spawns = MagicMock(return_value=np.repeat([[0, 0]], n_teams, axis=0))
    spg.generate = MagicMock(return_value=np.repeat([[0, 0]], n_agents, axis=0))
    return spg


def mock_ai():
    ai = Mock()
    ai.act = MagicMock()
    return ai
