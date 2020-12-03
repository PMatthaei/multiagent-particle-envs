import random

import numpy as np


# individual agent policy interface
from multiagent.environment import MultiAgentEnv


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
    def __init__(self, env: MultiAgentEnv, agent_index):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.agent_index = agent_index

    def action(self, obs):
        if self.env.discrete_action_input:
            agent = self.env.agents[self.agent_index]
            avail_actions = self.env.get_available_actions(agent)
            u = [random.choice(avail_actions)]
            # u = [random.randint(0, self.env.action_space[0].n - 1)]
        else:
            u = np.zeros(5)  # 5-d because of no-move action
            # generate random int from 0 to action space max index
            act_ind = random.randint(0, self.env.action_space[self.agent_index].n - 1)
            u[act_ind] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])


# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.comm = [False for i in range(env.world.dim_c)]

    def action(self, obs, move=None):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if move[0]: u = 1
            if move[1]: u = 2
            if move[2]: u = 4
            if move[3]: u = 3
        else:
            u = np.zeros(5)  # 5-d because of no-move action
            if move[0]: u[1] += 1.0
            if move[1]: u[2] += 1.0
            if move[3]: u[3] += 1.0
            if move[2]: u[4] += 1.0
            if True not in move:
                u[0] += 0.0
        action = np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return action
