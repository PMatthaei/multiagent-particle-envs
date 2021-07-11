from __future__ import annotations

import logging
import math
from enum import Enum, IntEnum

import numpy as np
import scipy.spatial.ckdtree
import scipy.spatial.distance
import random

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


class EntityState(object):
    def __init__(self):
        self.pos = None
        self.max_health = 0
        self.max_shield = 0
        self._health = []
        self.shield = 0

    @property
    def health(self):
        # This getter and setter setup is needed to convert the reference to the numpy array
        # Reference to numpy array values does not work
        return self._health[0]

    @health.setter
    def health(self, health):
        self._health[0] = health

    def reset(self):
        self._health = self.max_health
        self.shield = self.max_shield


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

    def reset(self):
        super().reset()
        self.c = None


class Action(object):
    def __init__(self, index=None, owner=None, target=None, u=None, c=None):
        self.index = index  # index associated with the action
        self.owner = owner  # id of agent who deployed action
        self.target = target  # target of the action
        self.u = u  # physical action
        self.c = c  # communication action


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        self.id = None
        self.target_id = None  # agent id of the current target
        self.attack_range = None  # attack range -> attacks available
        self.attack_damage = None
        self.sight_range = None  # observation range
        self.bounding_circle_radius = 6  # visuals and optional collision
        self.name = ''
        self.color = [0.75, 0, 0]
        self.state = EntityState()

    def is_alive(self):
        return self.state.health > 0

    def is_dead(self):
        is_dead = not self.is_alive()
        if is_dead:
            logger.debug("Agent {0} is dead.".format(self.id))
        return is_dead


class WorldObject(Entity):
    def __init__(self):
        super(WorldObject, self).__init__()


class Team:
    def __init__(self, tid, members, is_scripted=False):
        self.tid = tid
        self.members = members
        self.is_scripted = is_scripted
        self.size = len(members)  # team size not influenced by deaths


class PerformanceStatistics:
    def __init__(self, kills=0, assists=0, dmg_dealt=0, dmg_healed=0, attacks_performed=0, heals_performed=0,
                 distance_traveled=0, dmg_received=0):
        self.kills = kills
        self.assists = assists
        self.dmg_received = dmg_received
        self.dmg_dealt = dmg_dealt
        self.dmg_healed = dmg_healed
        self.attacks_performed = attacks_performed
        self.heals_performed = heals_performed
        self.distance_traveled = distance_traveled

    def reset(self):
        self.kills = 0
        self.assists = 0
        self.dmg_received = 0
        self.dmg_dealt = 0
        self.dmg_healed = 0
        self.attacks_performed = 0
        self.heals_performed = 0
        self.distance_traveled = 0


class Agent(Entity):
    def __init__(self, id, tid, color, build_plan, is_scripted=False):
        super(Agent, self).__init__()
        self.id = id
        # team id
        self.tid = tid
        self.is_scripted = is_scripted
        self.name = 'Agent %d' % id
        self.color = color
        self.unit_id = (build_plan['role'], build_plan['attack_type'])
        self.unit_type_bits = UNIT_TYPE_BITS[self.unit_id]
        self.unit_type_bits_n = len(self.unit_type_bits)
        self.attack_type = build_plan['attack_type']
        self.attack_data = self.attack_type.value
        self.role_type = build_plan['role']
        self.role_data = self.role_type.value

        self.attack_range = self.attack_data['attack_range']
        self.sight_range = self.attack_data['sight_range']
        assert self.sight_range > self.attack_range, "Sight range has to be greater than the attack range."
        self.attack_damage = self.role_data['attack_damage']

        self.state = AgentState()
        self.state.max_health = self.role_data['max_health']

        self.action = Action()
        self.stats = PerformanceStatistics()  # collects stats about the agent

    @property
    def self_observation(self):
        self_obs = [
                       self.state.health / self.state.max_health,  # relative health
                   ] + self.unit_type_bits
        return self_obs

    def heal(self, target: Agent):
        if target.tid != self.tid:  # Agents can not heal their enemies. This indicates a bug.
            raise IllegalTargetError(self)
        max_healed = target.state.health + self.attack_damage
        new_health = np.minimum(max_healed, target.state.max_health)
        healed = new_health - target.state.health
        target.state.health = new_health

        self.stats.dmg_healed += healed
        logger.debug("Agent {0} in team {1} healed Agent {2} in team {3} for {4}"
                     .format(self.id, self.tid, target.id, target.tid, self.attack_damage))

    def attack(self, other: Agent):
        if other.tid == self.tid:  # Agents can not attack their team mates. This indicates a bug.
            raise IllegalTargetError(self)
        other.state.health -= self.attack_damage
        logger.debug("Agent {0} in team {1} attacked Agent {2} in team {3} for {4}"
                     .format(self.id, self.tid, other.id, other.tid, self.attack_damage))
        self.stats.dmg_dealt += self.attack_damage
        other.stats.dmg_received += self.attack_damage
        if other.is_dead():
            self.stats.kills += 1
            logger.debug("Agent {0} is dead.".format(other.id))

    def has_heal(self):
        return 'can_heal' in self.role_data and self.role_data['can_heal']

    def can_heal(self, target=None):
        return self.has_heal() and (target is not None and target.tid == self.tid) \
               and target.is_alive() and target.state.health < target.state.max_health


class World(object):
    def __init__(self, grid_size: int, n_agents: int, n_teams: int, bounds=np.array([1280, 720]),
                 ai="basic", ai_config=None,
                 attack_range_only=True,
                 log=False):
        """
        Multi-agent world
        :param bounds: World bounds in which the agents can move
        """
        self.bounds = bounds
        self.log = log
        from maenv.ai import REGISTRY as ai_REGISTRY
        self.scripted_ai = ai_REGISTRY[ai](ai_config)
        self.positions = None
        self.grid_size = grid_size
        self.attack_range_only = attack_range_only
        # list of teams build by a subset of ...
        self.teams = []
        self.teams_n = n_teams
        self.wiped_teams = []
        # list of agents
        self.agents = []
        self.agents_n = n_agents
        # list of non-agent objects in the world
        self.objects = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3

        # Holds each agents alive boolean
        self.alive = np.zeros((n_agents,), dtype=int)
        # Team affiliation (team id) for later masking
        self.team_affiliations = np.full((n_agents,), -1, dtype=int)
        # Holds each agents health and max health
        self.health = np.zeros((n_agents,), dtype=float)
        self.max_health = np.zeros((n_agents,), dtype=int)
        # Holds each agents action
        self.actions = np.zeros((n_agents, self.dim_p + 1))
        # Holds all available movement actions in the current step - all moves are initially allowed if spawns are correct
        self.avail_movement_actions = np.ones((n_agents, self.get_movement_dims), dtype=float)  # 4 movement directions
        self.moves = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]) * self.grid_size  # W/E/N/S move

        # Holds all available target actions in the current step - all targets are blocked in the beginning
        self.avail_target_actions = np.zeros((n_agents, n_agents), dtype=float)  # target action for each agent
        # Mask out each agent if its himself
        self.self_target_mask = (np.ones_like(self.avail_target_actions) - np.diag(np.ones(self.agents_n))).astype(bool)
        # Mask all healable targets
        self.heal_target_mask = np.zeros_like(self.avail_target_actions).astype(bool)
        # Mask all attackable targets
        self.attack_target_mask = np.zeros_like(self.avail_target_actions).astype(bool)

        # Holds each agents sight range
        self.sight_ranges = np.zeros((n_agents,), dtype=float)
        # Holds each agents attack range
        self.attack_ranges = np.zeros((n_agents,), dtype=float)
        # Holds each agents unit representation encoded as bit array
        self.unit_bits_obs = np.zeros((n_agents, UNIT_BITS_NEEDED), dtype=float)
        # Holds each agents position in real and complex space
        self.positions = np.zeros((n_agents, self.dim_p))
        self.positions_c = np.zeros((1, n_agents), dtype=complex)
        # Holds all positions an agent can step on in the current state
        self.stepable_positions = np.zeros((n_agents, self.get_movement_dims, self.dim_p))
        # Holds each agents distance to other agents (and himself on diag = always 0)
        self.distances = np.full((n_agents, n_agents), fill_value=np.inf)
        # Holds each agents visibility of other agents (and himself on diag = always True)
        self.visibility = np.zeros((n_agents, n_agents))
        # Holds each agents visibility of other agents (and himself on diag = always True)
        self.reachability = np.zeros((n_agents, n_agents))
        # Holds each agents observation of all other agents
        self.obs = np.zeros((n_agents, n_agents, self.obs_dims))

        # Helper to calculate range queries
        self.kd_tree = None

        # Helper to generate points within the world
        self.spg = SpawnGenerator(self.grid_center, grid_size, self.dim_p, n_agents)

    def is_free(self, pos: np.array):
        """
        Checks is a given position is not occupied in the world and therefore free to move.
        This function is used during action selection.
        @param pos:
        @return:
        """
        pos_occupied = np.all(self.positions[self.alive == 1] == pos, axis=1)
        return not np.any(pos_occupied)

    def is_valid_move(self, agent_id: int):
        """
        Check if the given agent has made an illegal move because the state changed after action selection.
        @param agent_id:
        @return:
        """
        alive = self.alive[agent_id]  # save alive status
        self.alive[agent_id] = 0  # fake agent as death to prevent comparison with its own position in is_free
        pos = self.positions[agent_id]  # the position under testing
        valid = self.is_free(pos)
        self.alive[agent_id] = alive  # revert faked death
        return valid

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
    def alive_agents(self):
        return [agent for agent in self.agents if agent.is_alive()]

    @property
    def alive_scripted_agents(self):
        return [agent for agent in self.scripted_agents if agent.is_scripted and agent.is_alive()]

    @property
    def grid_center(self):
        center = self.bounds / 2.0
        center -= center % self.grid_size
        return center

    @property
    def center(self):
        return self.bounds[0] / 2.0, self.bounds[1] / 2.0

    @property
    def entities(self):
        return self.agents + self.objects

    @property
    def policy_teams(self):
        return [team for team in self.teams if not team.is_scripted]

    @property
    def scripted_teams(self):
        return [team for team in self.teams if team.is_scripted]

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if not agent.is_scripted]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.is_scripted]

    def can_attack(self, agent: Agent, target: Agent):
        if agent.has_heal():
            return False
        if target.tid == agent.tid:
            raise IllegalTargetError(agent)

        return self.reachability[agent.id][target.id]

    def init(self):
        # Initial KD Tree
        self.kd_tree = scipy.spatial.cKDTree(data=self.positions)
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
        # Calculate stepable positions for AI
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

        # Update KDTree after positions-update
        self.kd_tree = scipy.spatial.cKDTree(data=self.positions)

        # Re-Init
        self.init()

    def _calculate_wiped_teams(self):
        self.wiped_teams = [np.all(np.logical_not(self.alive[self.team_affiliations == t.tid])) for t in self.teams]

    def _calculate_stepable_pos(self):
        m = self.get_movement_dims
        n = self.agents_n
        self.stepable_positions = self.positions.repeat(m, axis=0).reshape(n, m, -1) + self.moves

    def _update_pos(self, agent):
        """
        Update position and re-calculate visibility AFTER performing influence actions such as attacks and heals.
        Otherwise these actions would be illegal although they were legal in the state when the action was selected.
        In case: an agent would move out of range with the following update
        @param agent:
        @return:
        """
        move_vector = agent.action.u[:2].copy()
        if np.any(move_vector):  # has movement
            pos = self.positions[agent.id]
            new_pos = pos + move_vector
            if self.is_free(new_pos):  # move is allowed
                self.positions[agent.id] += move_vector
                self.positions_c[0, agent.id] += complex(*move_vector)  # update complex position
            else:  # reset action if not allowed -> important to keep state consistent for rendering
                agent.action.u[:2] = 0.0

    def _update_visibility(self):
        self.visibility[:, :] = False  # Reset
        query = self.kd_tree.query_ball_point(self.positions, self.sight_ranges)
        visible = [[(agent, other, self.alive[agent]) for other in visibles] for agent, visibles in enumerate(query)]
        visible = np.array([item for sublist in visible for item in sublist])  # flatten
        xs, ys, alives = list(zip(*visible))  # Matrix coordinates and their corresponding value
        self.visibility[xs, ys] = alives  # If the agent is alive set its visible indices to True else False
        self.visibility[:, self.alive == 0] = False  # Set the visibility of all dead agents to False

    def _update_reachability(self):
        self.reachability[:, :] = False  # Reset
        query = self.kd_tree.query_ball_point(self.positions, self.attack_ranges)
        reachable = [[(agent, other, self.alive[agent]) for other in reachables] for agent, reachables in enumerate(query)]
        reachable = np.array([item for sublist in reachable for item in sublist])  # flatten
        xs, ys, alives = list(zip(*reachable))  # Matrix coordinates and their corresponding value
        self.reachability[xs, ys] = alives  # If the agent is alive set its reachable indices to True else False
        self.reachability[:, self.alive == 0] = False  # Set the reachability of all dead agents to False

    def _update_dist_matrix(self):
        self.distances = abs(self.positions_c.T - self.positions_c)  # abs in complex space is distance in real space

    def _calculate_obs(self):
        not_visible_mask = self.visibility == 0

        ranges = self.sight_ranges[:, np.newaxis]
        range_matrix = np.repeat(ranges, self.agents_n, axis=1)[:, :, np.newaxis]
        position_differences = (self.positions - self.positions[:, None])[..., :]

        relative_positions_obs = position_differences / range_matrix
        relative_positions_obs[not_visible_mask] = [0.0, 0.0]  # relative position to invisible agents set to 0,0

        relative_distances_obs = (self.distances / ranges)[:, :, np.newaxis]
        relative_distances_obs[not_visible_mask] = 0.0  # relative distance to invisible agents set to 0

        health_obs = np.repeat(self.health[:, np.newaxis], self.agents_n, axis=1)
        max_health = np.repeat(self.max_health[:, np.newaxis], self.agents_n, axis=1)
        health_obs /= max_health  # Normalize by max health
        health_obs = health_obs[:, :, np.newaxis]
        health_obs[not_visible_mask] = 0.0  # health of invisible agents set to 0

        others_unit_bits_obs = np.repeat([self.unit_bits_obs], self.agents_n, axis=0)
        others_unit_bits_obs[not_visible_mask] = UNIT_TYPE_BITS[UNKNOWN_TYPE]  # unit-bits of invisible agents = unknown

        self.obs = np.concatenate(
            (
                self.visibility[:, :, np.newaxis],
                health_obs,
                relative_positions_obs,
                relative_distances_obs,
                others_unit_bits_obs,
            ),
            axis=2
        )

    def connect(self, agent, spawn=None):
        """
        Connect an agent with the world. World data concerning the agent is referenced to keep up-to-date data such as
        health, position and the alive status.

        Static data such as sight range, max health and others are once set
        @param agent:
        @param spawn:
        @return:
        """
        self.health[agent.id] = agent.state.max_health  # Set initial health
        agent.state._health = self.health[agent.id:(agent.id + 1)]  # Connect agent health with world data storage

        self.positions[agent.id] = spawn  # Set initial position in real and complex space
        self.positions_c[0, agent.id] = complex(*spawn) if spawn is not None else complex()
        agent.state.pos = self.positions[agent.id]  # Connect agent position with world data storage

        self.alive[agent.id] = agent.is_alive()  # Set initial alive status - agents assumed to be dead in the beginning

        # Static data
        self.sight_ranges[agent.id] = (agent.attack_range if self.attack_range_only else agent.sight_range)* self.grid_size
        self.attack_ranges[agent.id] = agent.attack_range * self.grid_size
        self.max_health[agent.id] = agent.state.max_health
        self.unit_bits_obs[agent.id] = agent.unit_type_bits
        team_mates = [mate.id for mate in self.agents if mate.tid == agent.tid]
        self.heal_target_mask[agent.id][team_mates] = True if agent.has_heal() else False
        enemies = [enemy.id for enemy in self.agents if enemy.tid != agent.tid]
        self.attack_target_mask[agent.id][enemies] = False if agent.has_heal() else True
        self.team_affiliations[agent.id] = agent.tid

    def _update_alive_status(self):
        self.alive = self.health > 0

    def calculate_avail_movements_actions(self):
        self.avail_movement_actions[:, :] = 0  # Reset
        if self.bounds is not None:
            m_dims = self.get_movement_dims
            n = self.agents_n

            # Calculate stepable positions after position updates to provide info for avail movement calculation
            self._calculate_stepable_pos()

            legal_step_mask = np.ones((n, m_dims), dtype=bool)  # Marks legal moves
            # Stepped pos for every agents pos
            stepped_positions_n_agent = self.stepable_positions.repeat(n, axis=1).reshape(n, m_dims, n, -1)
            # np.all = pos overlap in x and y, np.any = any step overlap with any agent pos
            occupied_mask = np.any(np.all(stepped_positions_n_agent == self.positions, axis=3), axis=2)
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
        target_mask = (self.attack_target_mask | self.heal_target_mask)
        mask = (self.visibility == 1) & self.alive & self.self_target_mask & target_mask
        self.avail_target_actions[mask] = 1.0


