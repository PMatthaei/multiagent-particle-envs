import random

from maenv.environment import MAEnv


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
    def __init__(self, env: MAEnv, agent):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.agent = agent

    def action(self, obs=None):
        avail_actions = self.env.get_available_action_ids(self.agent)
        u = random.choice(avail_actions)
        return u
