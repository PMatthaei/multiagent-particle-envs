from __future__ import annotations

import logging
import math
from enum import Enum, IntEnum

import torch as th
import scipy.spatial.ckdtree
import scipy.spatial.distance
import random

from torch import Tensor

from maenv.exceptions.agent_exceptions import NoTargetFoundError, IllegalTargetError
from maenv.utils.spawn_generator import SpawnGenerator
from maenv.utils.unit_type_bit_encoder import unit_type_bits, bits_needed, UNKNOWN_TYPE

logger = logging.getLogger("ma-env")


class RoleTypes(Enum):
    TANK = {"max_health": 60, "attack_damage": 8, "id": 0}
    ADC = {"max_health": 40, "attack_damage": 20, "id": 1}
    HEALER = {"max_health": 40, "attack_damage": 10, "can_heal": True, "id": 2}

    def __int__(self):
        return self.value["id"]


class UnitAttackTypes(Enum):
    RANGED = {"attack_range": 3, "sight_range": 4}  # attack range units are in grid cells!
    MELEE = {"attack_range": 1, "sight_range": 2}


TYPES = [RoleTypes, UnitAttackTypes]
UNIT_BITS_NEEDED = bits_needed(TYPES)
UNIT_TYPE_BITS = unit_type_bits(TYPES)


class ActionTypes(IntEnum):
    NOOP = 0,
    WEST = 1,
    EAST = 2,
    SOUTH = 3,
    NORTH = 4,


class AF(IntEnum):  # AgentFeatures - Mapping for name to index in AgentTensor
    aid = 0,
    tid = 1,
    x = 2,
    y = 3,
    health = 4,
    max_health = 5,
    sight_range = 6,
    attack_range = 7,
    x_move = 8,
    y_move = 9,
    target = 10,
    alive = 11,
    has_heal = 12,


class World(object):
    def __init__(self, grid_size: int, n_agents: int, n_teams: int, bounds=th.tensor([1280, 720]),
                 ai="basic", ai_config=None,
                 attack_range_only=True,
                 log=False):
        """
        Multi-agent world
        :param bounds: World bounds in which the agents can move
        """
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_teams = n_teams
        self.bounds = bounds
        from maenv.ai import REGISTRY as ai_REGISTRY
        self.scripted_ai = ai_REGISTRY[ai](ai_config)
        self.attack_range_only = attack_range_only
        self.log = log

        self.dim_p = 2
        self.agents = th.zeros(self.n_agents, self.n_features)
        self.moves = th.tensor([[-1, 0], [1, 0], [0, 1], [0, -1]])

        self.visible = th.zeros(self.n_agents, self.n_agents)
        self.attackable = th.zeros(self.n_agents, self.n_agents)
        self.avail_movement_actions = th.ones((n_agents, self.get_movement_dims), dtype=th.float16)
        self.avail_target_actions = th.zeros((n_agents, n_agents), dtype=th.float16)

        # Helper to calculate range queries
        self.kd_tree = None

        # Helper to generate points within the world
        self.spg = SpawnGenerator(self.grid_center, grid_size, self.dim_p, n_agents)

    @property
    def n_features(self):
        return len(AF)

    @property
    def get_movement_dims(self):
        return self.dim_p * 2

    @property
    def obs_dims(self):
        obs_dims = self.dim_p  # position
        obs_dims += 1  # visibility bool
        obs_dims += 1  # distance
        obs_dims += 1  # health
        obs_dims += UNIT_BITS_NEEDED
        return obs_dims

    @property
    def grid_center(self):
        center = self.bounds / 2.0
        center -= center % self.grid_size
        return center

    @property
    def center(self):
        return self.bounds[0] / 2.0, self.bounds[1] / 2.0

    def is_free(self, pos: Tensor):
        """
        Checks is a given position is not occupied in the world and therefore free to move.
        This function is used during action selection.
        @param pos:
        @return:
        """
        positions = self.agents[:, [2, 3]]
        alive = self.agents[:, [4, 5]]
        pos_occupied = th.all(positions[alive == 1] == pos, dim=1)
        return not th.any(pos_occupied)

    def is_valid_move(self, agent_id: int):
        """
        Check if the given agent has made an illegal move because the state changed after action selection.
        @param agent_id:
        @return:
        """
        alive = self.agents[agent_id, [4, 5]]  # save alive status
        self.agents[agent_id, [4, 5]] = 0  # fake agent as death to prevent comparison with its own position in is_free
        pos = self.agents[agent_id, [2, 3]]  # the position under testing
        valid = self.is_free(pos)
        self.agents[agent_id, [4, 5]] = alive  # revert faked death
        return valid

    def can_attack(self, aid: int, target: int):
        if self.agents[aid, AF.has_heal]:
            return False
        if self.agents[aid, AF.tid] == self.agents[target, AF.tid]:
            raise IllegalTargetError(aid)

        return self.attackable[aid, target]

    def init(self):
        # Update KDTree after positions-update
        positions = self.agents[:, [2, 3]]
        self.kd_tree = scipy.spatial.cKDTree(data=positions)
        #
        # End of state transition - Calculate observations
        #
        self._update_visibility()  # Used for obs-calculation

        self._update_reachability()

        self._update_dist_matrix()  # Used for obs-calculation

        self._calculate_obs()  # Calculate what is observed after the step is calculated

        #
        # Calculate each agents available actions after the step is calculated
        #
        self.calculate_avail_movements_actions()

        self.calculate_avail_target_actions()

        # After state transition test win condition
        self._calculate_wiped_teams()

    def step(self):
        """
        Update state of the world.
        """
        # Calculate stepable positions for AI --> Used for upcoming act() calls
        self._calculate_stepable_pos()

        # Set actions for scripted/heuristic agents BEFORE advancing state
        for scripted_agent in self.alive_scripted_agents:
            self.scripted_ai.act(scripted_agent, self)

        # Shuffle randomly to prevent favoring
        # Calculate influence actions BEFORE updating positions to prevent moving out of range after action was set
        for agent in random.sample(self.alive_agents, len(self.alive_agents)):
            # Influence entity if target set f.e with attack, heal etc
            agent_has_action_target = agent.action.u[2] != -1
            if agent_has_action_target:
                agent.target_id = int(agent.action.u[2])
                if agent.target_id is None or math.isnan(agent.target_id):
                    raise NoTargetFoundError()
                target = self.agents[agent.target_id]
                if agent.can_heal(target):
                    agent.heal(target)
                elif self.can_attack(agent, target):
                    agent.attack(target)
                else:
                    if self.log:
                        logger.debug(
                            "Agent {0} cannot attack Agent {1} due to range.".format(agent.id, agent.target_id))

                agent.target_id = None  # Reset target after processing

        # Update alive status BEFORE moving the agents
        self._update_alive_status()

        # Update positions BEFORE recalculating visibility and observations
        for a in random.sample(self.alive_agents, len(self.alive_agents)):
            self._update_pos(a)

        # Re-Init
        self.init()

    def _calculate_wiped_teams(self):
        team_affiliations = self.agents[:, [1]]
        alive = self.agents[:, [4, 5]]
        self.wiped_teams = [th.all(th.logical_not(alive[team_affiliations == tid])) for tid in range(self.n_teams)]

    def _calculate_stepable_pos(self):
        m = self.get_movement_dims
        n = self.n_agents
        positions = self.agents[:, [2, 3]]
        self.stepable_positions = positions.repeat(repeats=(m,)).reshape(n, m, -1) + self.moves  # TODO

    def _update_pos(self, agent_id: int):
        """
        Update position and re-calculate visibility AFTER performing influence actions such as attacks and heals.
        Otherwise these actions would be illegal although they were legal in the state when the action was selected.
        In case: an agent would move out of range with the following update
        @param agent:
        @return:
        """
        positions = self.agents[:, [2, 3]]
        move_vector = self.agents[:, [8, 9]]
        if th.any(move_vector):  # has movement
            pos = positions[agent_id]
            new_pos = pos + move_vector
            if self.is_free(new_pos):  # move is allowed
                positions[agent_id] += move_vector
            else:  # reset action if not allowed -> important to keep state consistent for rendering
                self.agents[agent_id, [8, 9]] = 0  # Reset action

    def _update_visibility(self):
        self.visible[:, :] = False  # Reset
        positions = self.agents[:, [2, 3]]
        sight_ranges = self.agents[:, [6]]
        query = self.kd_tree.query_ball_point(positions, sight_ranges)
        visible = [[(agent, other, self.alive[agent]) for other in visibles] for agent, visibles in enumerate(query)]
        visible = th.tensor([item for sublist in visible for item in sublist])  # flatten
        xs, ys, alives = list(zip(*visible))  # Matrix coordinates and their corresponding value
        self.visible[xs, ys] = alives  # If the agent is alive set its visible indices to True else False
        self.visible[:, self.alive == 0] = False  # Set the visibility of all dead agents to False

    def _update_reachability(self):
        self.attackable[:, :] = False  # Reset
        positions = self.agents[:, [2, 3]]
        attack_ranges = self.agents[:, [7]]
        query = self.kd_tree.query_ball_point(positions, attack_ranges)
        reachable = [[(agent, other, self.alive[agent]) for other in reachables] for agent, reachables in
                     enumerate(query)]
        reachable = th.tensor([item for sublist in reachable for item in sublist])  # flatten
        xs, ys, alives = list(zip(*reachable))  # Matrix coordinates and their corresponding value
        self.attackable[xs, ys] = alives  # If the agent is alive set its reachable indices to True else False
        self.attackable[:, self.alive == 0] = False  # Set the reachability of all dead agents to False

    def _update_dist_matrix(self):
        positions = self.agents[:, [2, 3]]
        self.distances = th.cdist(positions, positions, p=2.0)

    def _calculate_obs(self):
        not_visible_mask = self.visible == 0
        positions = self.agents[:, [2, 3]]
        sight_ranges = self.agents[:, [6]]
        health = self.agents[:, [4]]
        max_health = self.agents[:, [5]]

        ranges = sight_ranges[:, th.newaxis]
        range_matrix = th.repeat(ranges, self.n_agents, axis=1)[:, :, th.newaxis]
        position_differences = (positions - positions[:, None])[..., :]

        relative_positions_obs = position_differences / range_matrix
        relative_positions_obs[not_visible_mask] = [0.0, 0.0]  # relative position to invisible agents set to 0,0

        relative_distances_obs = (self.distances / ranges)[:, :, th.newaxis]
        relative_distances_obs[not_visible_mask] = 0.0  # relative distance to invisible agents set to 0

        health_obs = th.repeat(health[:, th.newaxis], self.n_agents, axis=1)
        max_health = th.repeat(max_health[:, th.newaxis], self.n_agents, axis=1)
        health_obs /= max_health  # Normalize by max health
        health_obs = health_obs[:, :, th.newaxis]
        health_obs[not_visible_mask] = 0.0  # health of invisible agents set to 0

        others_unit_bits_obs = th.repeat([self.unit_bits_obs], self.n_agents, axis=0)
        others_unit_bits_obs[not_visible_mask] = UNIT_TYPE_BITS[UNKNOWN_TYPE]  # unit-bits of invisible agents = unknown

        self.obs = th.cat(
            (
                self.visible[:, :, th.newaxis],
                health_obs,
                relative_positions_obs,
                relative_distances_obs,
                others_unit_bits_obs,
            ),
            dim=2
        )

    def _update_alive_status(self):
        self.agents[:, [AF.alive]] = self.agents[:, [AF.health]] > 0

    def calculate_avail_movements_actions(self):
        self.avail_movement_actions[:, :] = 0  # Reset
        if self.bounds is not None:
            m_dims = self.get_movement_dims
            n = self.n_agents

            # Calculate stepable positions after position updates to provide info for avail movement calculation
            self._calculate_stepable_pos()

            legal_step_mask = th.ones((n, m_dims), dtype=th.bool)  # Marks legal moves
            # Stepped pos for every agents pos
            stepped_positions_n_agent = self.stepable_positions.repeat(n, dim=1).reshape(n, m_dims, n, -1)
            # th.all = pos overlap in x and y, th.any = any step overlap with any agent pos
            occupied_mask = th.any(th.all(stepped_positions_n_agent == self.agents[:, [AF.x, AF.y]], dim=3), dim=2)
            legal_step_mask[occupied_mask] = False  # Mask contains entries which are occupied

            # In bounds checks
            x_in_left_bound = self.stepable_positions[:, :, 0] >= 0
            x_in_right_bound = self.stepable_positions[:, :, 0] <= self.bounds[0]
            y_in_down_bound = self.stepable_positions[:, :, 1] >= 0
            y_in_up_bound = self.stepable_positions[:, :, 1] <= self.bounds[1]
            all_in_bound = (x_in_left_bound & x_in_right_bound) & (y_in_up_bound & y_in_down_bound)
            mask = legal_step_mask & all_in_bound

            self.avail_movement_actions[mask] = 1.0
        else:  # unbounded map -> always add all_in_bound movement directions
            self.avail_movement_actions[:, :] = 1.0

    def calculate_avail_target_actions(self):
        self.avail_target_actions[:, :] = 0.0  # Reset
        target_mask = self.attack_target_mask | self.heal_target_mask
        alive = th.unsqueeze(self.agents[:, [AF.alive]], dim=1)
        self.avail_target_actions = (self.attackable == 1) & alive & self.self_target_mask & target_mask
