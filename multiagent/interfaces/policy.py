import random

import numpy as np

# individual agent policy interface
from multiagent.environment import MAEnv


class Policy(object):
    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()


#
# Example/Test policies
#
# random agent policy
class RandomPolicy(Policy):
    def __init__(self, env: MAEnv, agent_index):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.agent_index = agent_index

    def action(self, obs):
        agent = self.env.agents[self.agent_index]
        avail_actions = self.env.get_available_action_ids(agent)
        u = random.choice(avail_actions)
        return u
