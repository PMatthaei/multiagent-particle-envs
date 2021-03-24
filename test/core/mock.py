from unittest.mock import MagicMock, Mock

import numpy as np

from multiagent.core import Team


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
    team = Team(tid, members, is_scripted)
    return team
