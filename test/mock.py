from unittest.mock import MagicMock, Mock

import numpy as np

from multiagent.core import Team, World


def mock_agent(id: int, tid: int = 0, sight_range=2):
    agent = Mock()
    agent.id = id
    agent.tid = tid
    agent.state.max_health = 100
    agent.state.health = 100
    agent.is_alive = MagicMock(return_value=True)
    agent.sight_range = sight_range
    agent.has_heal = MagicMock(return_value=False)
    agent.action.u = np.zeros((2,))
    agent.unit_type_bits = [0, 0, 1]
    return agent


def mock_team(tid: int, members=None, is_scripted=False):
    if members is None:
        members = []
    team = Mock(tid=tid, members=members, is_scripted=is_scripted)
    return team


def mock_world(agents_n, grid_size=10, teams=None):
    if teams is None:
        teams = []
    world = Mock(agents_n=agents_n, grid_size=grid_size)
    world.teams = teams
    return world


def mock_spawn_generator(n_teams, n_agents):
    spg = Mock()
    spg.generate_team_spawns = MagicMock(return_value=np.repeat([[0, 0]], n_teams, axis=0))
    spg.generate = MagicMock(return_value=np.repeat([[0, 0]], n_agents, axis=0))
    return spg
