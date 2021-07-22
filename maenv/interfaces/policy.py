import random

from core import Agent
from maenv.environment import MAEnv


class Policy(object):
    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()


class RandomPolicy(Policy):
    def __init__(self, env: MAEnv, agent: Agent):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.agent = agent

    def action(self, obs=None):
        avail_actions = self.env.get_available_action_ids(self.agent)
        print(f"Agent {self.agent.id} can choose: {avail_actions}")
        u = random.choice(avail_actions)
        print(f"Agent {self.agent.id} chose {u}")
        return u
