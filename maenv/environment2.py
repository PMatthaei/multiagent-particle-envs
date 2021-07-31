import logging
import random
from enum import IntEnum, Enum
from typing import List, Dict

import gym
from gym import spaces

from utils import bits_needed, unit_type_bits


class UnitRoleTypes(Enum):
    TANK = {"max_health": 60, "attack_damage": 8, "id": 0}
    ADC = {"max_health": 40, "attack_damage": 20, "id": 1}
    HEALER = {"max_health": 40, "attack_damage": 10, "can_heal": True, "id": 2}

    def __int__(self):
        return self.value["id"]


class UnitAttackTypes(Enum):
    RANGED = {"attack_range": 3, "sight_range": 4, "id": 0}  # attack range units are in grid cells!
    MELEE = {"attack_range": 1, "sight_range": 2, "id": 1}


class Actions(IntEnum):
    NOOP = 0,
    WEST = 1,
    EAST = 2,
    SOUTH = 3,
    NORTH = 4,


plan = {
    "teams": {
        "home": {
            "tid": 0,
            "agents": [
                {
                    "role": UnitRoleTypes.TANK,
                    "attack": UnitAttackTypes.RANGED
                },
                {
                    "role": UnitRoleTypes.TANK,
                    "attack": UnitAttackTypes.RANGED
                }
            ],
            "is_scripted": False
        },
        "adversary": {
            "tid": 1,
            "agents": [
                {
                    "role": UnitRoleTypes.TANK,
                    "attack": UnitAttackTypes.RANGED
                },
                {
                    "role": UnitRoleTypes.TANK,
                    "attack": UnitAttackTypes.RANGED
                }
            ],
            "is_scripted": True
        }
    }
}


class AF(IntEnum):  # AgentFeatures - Mapping for name to index in AgentTensor
    aid = 0,
    tid = 1,
    x = 2,
    y = 3,
    health = 4,
    max_health = 5,
    sight_range = 6,
    attack_range = 7,
    damage = 8,
    x_move = 9,
    y_move = 10,
    target = 11,
    alive = 12,
    has_heal = 13,


class OF(IntEnum):  # ObservationFeatures - Mapping for name to index in AgentTensor
    x = 2,
    y = 3,
    health = 4,
    alive = 5,
    visible = 5


class SOF(IntEnum):  # SelfObservationFeatures - Mapping for name to index in AgentTensor
    x = 2,
    y = 3,
    health = 4,
    alive = 5,
    visible = 5


class Agent:
    def __int__(self, role: UnitRoleTypes, attack: UnitAttackTypes):
        self.role = role
        self.attack = attack


class MultiAgentTeam:
    def __init__(self, tid: int, agents: List[Dict], is_scripted=False):
        self.tid = tid
        self.agents = [Agent(**agent) for agent in agents]
        self.is_scripted: bool = is_scripted


class MultiAgentPlan:
    def __init__(self, teams: List[Dict]):
        self.teams: List[MultiAgentTeam] = [MultiAgentTeam(**team) for team in teams]


class MultiAgentEnv(gym.Env):

    def __init__(self, grid_size: int):
        self.grid_size = grid_size

        self.logger = logging.getLogger("ma-env")
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.DEBUG)

        self.policy_agents = [1, 2, 3]
        self.scripted_agents = [4, 5, 6]
        self.n = len(self.policy_agents) + len(self.scripted_agents)
        self.t = 0
        self.episode = 0
        self.episode_limit = 60

        self.unit_type_bits_needed = bits_needed([UnitRoleTypes, UnitAttackTypes])
        self.unit_type_bits = unit_type_bits([UnitRoleTypes, UnitAttackTypes])

        self.feature_dim = len(AF)
        self.obs_dim = len(OF)  # Observation of others
        self.self_obs_dim = len(SOF)  # Self Observation
        self.move_dim = 2
        self.move_options = 2
        self.action_space = []
        self.observation_space = []

        self.state_dim = self._state_dim()
        for agent in self.policy_agents:
            self.action_space.append(spaces.Discrete(self._action_dim(agent)))
            self.observation_space.append(spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim(agent),), dtype=float))

        self.device = "cuda:0"

        self.agents = th.zeros(self.n, self.feature_dim, dtype=th.float32, device=self.device)
        self.moves = th.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=th.float32, device=self.device)
        self.visible = th.zeros(n, self.feature_dim, device=self.device)
        self.attackable = th.zeros(n, n, dtype=th.float32, device=self.device)
        self.avail_movement_actions = th.ones((n, self.moves.shape[0]), dtype=th.float32, device=self.device)
        self.avail_target_actions = th.zeros((n, n), dtype=th.float32, device=self.device)

        self.action_indices = th.zeros(self.n, device=self.device, dtype=th.int)

        # Fake set actions
        self.fake_action = th.tensor([[11, 22, 1]] * self.n, dtype=th.float32, device=self.device)
        # Fake damage of agents
        self.agents[:, [AF.damage]] = 99

        noop_action = th.tensor([[0, 0, 0]], device=self.device)
        # Initialize actions with movement actions
        movement_actions = th.tensor([
            [0, 0, 0],  # No move X, no move Y, no target -> No-Op = 0
            [-1, 0, 0],  # Move -X, no move Y, no target -> WEST = 1
            [1, 0, 0],  # Move +X, no move Y, no target -> EAST = 1
            [1, 0, 0],  # No move X, move +Y, no target -> NORTH = 1
            [-1, 0, 0],  # No move X, move -Y, no target -> SOUTH = 1
        ], device=self.device)

        self.fixed_actions = movement_actions.shape[0]
        movement_actions *= self.grid_size
        # Build target actions
        target_actions = th.zeros(self.n, 3, device=self.device)
        target_actions[:, 2] = th.range(self.fixed_actions, self.fixed_actions + self.n - 1)
        # Combine actions
        self.action_values = th.cat((noop_action, movement_actions, target_actions), dim=0)

    def step(self, actions):
        # TODO: fill actions with AI actions
        self.action_indices[:] = th.as_tensor(actions)
        self.logger.debug(f"Action: \n {self.action_indices}")

        n_info = {"battle_won": [], "draw": False}
        n_obs = []
        reward_n = []
        n_done = []

        # Set action values changes from action indices
        self.agents[:, [AF.x_move, AF.y_move, AF.target]] = th.index_select(self.action_values, 0, self.action_indices)

        # Update position
        self.agents[:, [AF.x, AF.y]] += self.agents[:, [AF.x_move, AF.y_move]]
        position_collision = self.agents[:, [AF.x, AF.y]] == self.agents[:, [AF.x, AF.y]]
        # TODO: which updates should be reverted -> choose one random agent of set of agents that share a illegal pos
        revert_mask = th.where(th.all(position_collision, dim=1))
        revert_mask = revert_mask[0].view(-1, 1)
        # Revert illegal positions randomly
        self.agents[revert_mask, [AF.x, AF.y]] -= self.agents[revert_mask, [AF.x_move, AF.y_move]]

        # TODO: Update visibility
        self.visible[:] = False

        target_mask = th.clamp(self.agents[revert_mask, [AF.target]] - self.fixed_actions, min=0)
        target_mask = target_mask.squeeze().to(dtype=th.int)

        # Apply attack
        attack_mask = th.bincount(target_mask)
        attack_mask = th.cat((attack_mask, th.zeros(self.n - attack_mask.shape[0], device=self.device)), dim=0)
        attack_mask = attack_mask.view(-1, 1)  # How many attacks against which id have been issued
        # TODO: Detect denied attacks because enemy already dead -> Detect overkill, how much overkill, deny random agents
        self.agents[:, [AF.health]] -= (attack_mask * self.agents[:, [AF.damage]])

        # Apply heal
        # TODO: How many heals have been issued against an agent
        self.agents[:, [AF.health]] += (attack_mask * self.agents[:, [AF.damage]])

        # TODO: Reward and calc terminate
        # Limit reached
        if self.episode_limit is not None and self.episode_limit == self.t:
            n_info["draw"] = True
        self.episode += 1
        n_done = [True] * len(n_done)

        self.t += 1

        return n_obs, reward_n, n_done, n_info

    def transition(self):
        pass

    def reward(self, agent):
        pass

    def get_state(self):
        pass

    def get_obs(self):
        pass

    def get_env_info(self):
        return {
            "state_shape": self.state_dim,
            "obs_shape": self.observation_space[0].shape[0],
            "n_actions": self.action_space[0].n,
            "n_agents": self.n,
            "episode_limit": self.episode_limit
        }

    def get_available_actions(self, agent):
        pass

    def _state_dim(self):
        state_dim = 0
        for _ in range(self.n):
            state_dim += self.obs_dim + self.move_dim
        return state_dim

    def _action_dim(self, agent):
        movement_dims = self.move_dim * self.move_options  # two options for every movement axis
        return movement_dims + self.n + 1  # no-op

    def _obs_dim(self, agent):
        movement_dims = self.move_dim * self.move_options  # two options for every movement axis
        return movement_dims + self.n + 1  # no-op

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    import sys
    import torch as th

    n = 10
    env = MultiAgentEnv(grid_size=10)

    obs_n = env.reset()
    env.render()

    try:
        while True:
            # query for action from each agent's policy
            act_n = [random.randint(0, actions.n) for actions in env.action_space] * 2  # KI copies agent
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            state = env.get_state()
            # render all agent views
            env.render()

            if any(done_n):
                env.reset()
    except KeyboardInterrupt:
        env.close()
        sys.exit()
