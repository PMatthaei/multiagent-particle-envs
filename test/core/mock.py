from unittest.mock import MagicMock, Mock

import numpy as np


def mock_agent(id: int, tid: int = 0):
    agent = Mock()
    agent.id = id
    agent.tid = tid
    agent.state.max_health = 100
    agent.is_alive = MagicMock(return_value=True)
    agent.sight_range = 2
    agent.action.u = np.zeros((2,))
    return agent
