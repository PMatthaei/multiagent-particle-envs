from __future__ import annotations

import itertools
import logging
import math
from enum import Enum, IntEnum

import numpy as np
import scipy.spatial.ckdtree
import scipy.spatial.distance

from multiagent.exceptions.agent_exceptions import NoTargetFoundError, IllegalTargetError, OverhealError
from multiagent.exceptions.world_exceptions import NoTeamFoundError

logger = logging.getLogger("ma-env")

UNKNOWN_TYPE = "UNIT_TYPE_NONE"


class RoleTypes(Enum):
    TANK = {"max_health": 60, "attack_damage": 10}
    ADC = {"max_health": 40, "attack_damage": 20}
    HEALER = {"max_health": 60, "attack_damage": 5, "can_heal": True}


class UnitAttackTypes(Enum):
    RANGED = {"attack_range": 35}
    MELEE = {"attack_range": 5}


# Calculate all unique unit types
UNIQUE_UNIT_TYPES = list(itertools.product(RoleTypes, UnitAttackTypes))
UNIQUE_UNIT_TYPES.insert(0, UNKNOWN_TYPE)
# Calculate bits needed to represent all unique units
UNIT_BITS_NEEDED = math.ceil(math.log(len(UNIQUE_UNIT_TYPES), 2))


def _to_bits(num):
    return list(map(float, bin(num)[2:].zfill(UNIT_BITS_NEEDED)))


UNIT_TYPE_BITS: dict = dict((unit, _to_bits(index)) for index, unit in enumerate(UNIQUE_UNIT_TYPES))


class ActionTypes(IntEnum):
    NOOP = 0,
    WEST = 1,
    EAST = 2,
    SOUTH = 3,
    NORTH = 4,


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        self.pos = None
        self.max_health = 0
        self.max_shield = 0
        self._health = []
        self.shield = 0

    @property
    def health(self):
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
        """
        # Action of the agent
        :param index:
        :param owner:
        :param target:
        :param u: list containing all physical actions with the following assignment:
        - Index 0: X-Axis direction (multiplied with step size)
        - Index 1: Y-Axis direction (multiplied with step size)
        - Index 2: Action target id (agent id)
        :param c:
        """
        # index associated with the action
        self.index = index
        # id of agent who deployed action
        self.owner = owner
        # target of the action
        self.target = target
        # physical action
        self.u = u
        # communication action
        self.c = c


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        self.id = None
        # ID of the current target
        self.target_id = None
        # how far can an entity attack
        self.attack_range = 15
        self.attack_damage = 20
        # how far can the entity see
        self.sight_range = 25
        # radius defines entity`s collision and visuals
        self.bounding_circle_radius = 3
        self.name = ''
        # entity can move / be pushed
        self.movable = False
        # color
        self.color = [0.75, 0, 0]

        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    def is_alive(self):
        return self.state.health > 0

    def is_dead(self):
        is_dead = not self.is_alive()
        if is_dead:
            logger.debug("Agent {0} is dead.".format(self.id))
        return is_dead

    @property
    def mass(self):
        return self.initial_mass


# properties of world objects
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


# properties of agent entities
class Agent(Entity):
    def __init__(self, id, tid, color, build_plan, action_callback=None):
        super(Agent, self).__init__()
        self.id = id
        # team id
        self.tid = tid
        self.name = 'Agent %d' % id
        self.color = color
        self.unit_id = (build_plan['role'], build_plan['attack_type'])
        self.unit_type_bits = UNIT_TYPE_BITS[self.unit_id]
        self.unit_type_bits_n = len(self.unit_type_bits)
        self.attack_type = build_plan['attack_type'].value
        self.role = build_plan['role'].value

        self.attack_range = self.attack_type['attack_range']
        self.sight_range = self.attack_range
        assert self.sight_range >= self.attack_range, "Sight range cannot be smaller than attack range."
        self.attack_damage = self.role['attack_damage']

        self.movable = True
        # control range
        # state
        self.state = AgentState()
        self.state.max_health = self.role['max_health']

        # action
        self.action = Action()
        # stats
        self.stats = PerformanceStatistics()
        # script behavior to execute
        self.action_callback = action_callback

    @property
    def self_observation(self):
        self_obs = [
                       self.state.health / self.state.max_health,  # relative health
                   ] + self.unit_type_bits
        return self_obs

    def heal(self, target: Agent):
        if target.tid != self.tid:  # Agents can not heal their enemies. This indicates a bug.
            raise IllegalTargetError(self)
        target.state.health += self.attack_damage
        if target.state.health > target.state.max_health:
            raise OverhealError(self)
        self.stats.dmg_healed += self.attack_damage
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
        return 'can_heal' in self.role and self.role['can_heal']

    def can_heal(self, target=None):
        return self.has_heal() and (target is not None and target.tid == self.tid) \
               and target.is_alive() and target.state.health < target.state.max_health


class World(object):
    def __init__(self, grid_size: int, agents_n: int, bounds=np.array([1280, 720]), log=False):
        """
        Multi-agent world
        :param bounds: World bounds in which the agents can move
        """
        self.bounds = bounds
        self.log = log
        self.positions = None
        self.grid_size = grid_size
        # indicates if the reward will be global(cooperative) or local
        self.collaborative = False
        # list of teams build by a subset of ...
        self.teams = []
        self.wiped_teams = []
        # list of agents
        self.agents = []
        self.agents_n = agents_n
        # list of non-agent objects in the world
        self.objects = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3

        # Holds each agents alive boolean
        self.alive = np.zeros((agents_n,), dtype=np.int)
        # Team mask
        self.team_mask = np.full((agents_n,), -1, dtype=np.int)
        # Holds each agents health and max health
        self.health = np.zeros((agents_n,), dtype=np.float)
        self.max_health = np.zeros((agents_n,), dtype=np.int)
        # Holds each agents action
        self.actions = np.zeros((agents_n, self.dim_p + 1))
        # Holds all available movement actions in the current step
        self.avail_movement_actions = np.ones((agents_n, self.get_movement_dims),
                                              dtype=np.float)  # four movement directions
        # Holds all available target actions in the current step
        self.avail_target_actions = np.zeros((agents_n, agents_n), dtype=np.float)  # target action for each agent
        # Mask out each agent if its himself
        self.self_target_mask = (np.ones_like(self.avail_target_actions) - np.diag(np.ones(self.agents_n))) \
            .astype(np.bool)
        # Mask all healable targets
        self.heal_target_mask = np.zeros_like(self.avail_target_actions).astype(np.bool)
        # Mask all attackable targets
        self.attack_target_mask = np.zeros_like(self.avail_target_actions).astype(np.bool)

        # Holds each agents sight range
        self.ranges = np.zeros((agents_n,), dtype=np.float)
        # Holds each agents unit representation encoded as bit array
        self.unit_bits_obs = np.zeros((agents_n, UNIT_BITS_NEEDED), dtype=np.float)
        # Holds each agents position
        self.positions = np.zeros((agents_n, self.dim_p))
        # Holds each agents distance to other agents (and himself on diag = always 0)
        self.distances = np.zeros((agents_n, agents_n))
        # Holds each agents visibility of other agents (and himself on diag = always True)
        self.visibility = np.zeros((agents_n, agents_n))
        # Holds each agents observation of all other agents
        self.obs = np.zeros((agents_n, agents_n, self.obs_dims))

        # Helper to calculate range queries
        self.kd_tree = scipy.spatial.kdtree.cKDTree(self.positions)

    def get_team(self, tid):
        for team in self.teams:
            if team.tid == tid:
                return team
        raise NoTeamFoundError(tid)

    def get_opposing_teams(self, tid: int):
        return [team for team in self.teams if team.tid != tid]

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
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def can_attack(self, agent: Agent, target: Agent):
        if target.tid == agent.tid:
            raise IllegalTargetError("Cannot attack team mates.")
        return self.visibility[agent.id][target.id]

    def step(self):
        """
        Update state of the world.
        """
        # Set actions for scripted/heuristic agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # Shuffle randomly to prevent favoring
        import random
        shuffled_agents = self.alive_agents.copy()
        random.shuffle(shuffled_agents)

        # Calculate influence actions BEFORE updating positions to prevent moving out of range
        for agent in shuffled_agents:
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
        for agent in shuffled_agents:
            self._update_pos(agent)

        self._update_visibility()

        self._update_dist_matrix()

        self._calculate_obs()

        self.calculate_avail_movements_actions()

        self.calculate_avail_target_actions()

        # After update test if world is done aka only one team left
        self._calculate_wiped_teams()

    def _calculate_wiped_teams(self):
        self.wiped_teams = [np.all(np.logical_not(self.alive[self.team_mask == team.tid])) for team in self.teams]

    def _update_pos(self, agent):
        """
        Update position and re-calculate visibility AFTER performing influence actions such as attacks and heals.
        Otherwise these actions would be illegal although they were legal in the state when the action was selected.
        In case: an agent would move out of range with the following update
        @param agent:
        @return:
        """
        move_vector = agent.action.u[:2]
        if not any(move_vector):
            return  # no movement
        pos = self.positions[agent.id, :2]
        new_pos = pos + move_vector
        if self.is_free(new_pos):
            pos += move_vector

    def _update_visibility(self):
        self.kd_tree = scipy.spatial.cKDTree(data=self.positions)
        self.visibility[:, :] = False  # Reset
        visible = self.kd_tree.query_ball_point(self.positions, self.ranges)
        for agent_id, visible_indices in enumerate(visible):
            self.visibility[agent_id, visible_indices] = self.alive[agent_id]
            dead = self.alive == 0
            self.visibility[agent_id, dead] = False

    def _update_dist_matrix(self):
        z = np.array([complex(*pos) for pos in self.positions])
        m, n = np.meshgrid(z, z)
        self.distances = abs(m - n)  # abs in complex space is distance in real space

    def _calculate_obs(self):
        not_visible_mask = self.visibility == 0

        ranges = np.repeat(self.ranges.reshape(-1, 1), self.agents_n, axis=1)
        position_differences = (self.positions - self.positions[:, None])[..., :]

        relative_positions_obs = position_differences / ranges.reshape(*ranges.shape, 1)
        relative_positions_obs[not_visible_mask] = [0.0, 0.0]  # relative position to invisible agents set to 0,0

        relative_distances_obs = (self.distances / self.ranges[:, None])
        relative_distances_obs = relative_distances_obs.reshape(*relative_distances_obs.shape, 1)
        relative_distances_obs[not_visible_mask] = 0.0  # relative distance to invisible agents set to 0

        visibility_obs = self.visibility.reshape(*self.visibility.shape, 1)

        health_obs = np.repeat(self.health.reshape(*self.health.shape, 1), self.agents_n, axis=1)
        max_health = np.repeat(self.max_health.reshape(*self.max_health.shape, 1), self.agents_n, axis=1)
        health_obs /= max_health
        health_obs = health_obs.reshape((*health_obs.shape, 1))
        health_obs[not_visible_mask] = 0.0  # health of invisible agents set to 0

        unit_bits_obs = np.repeat([self.unit_bits_obs], self.agents_n, axis=0)
        unit_bits_obs[not_visible_mask] = UNIT_TYPE_BITS[UNKNOWN_TYPE]  # unit bits of invisible agents set to unknown

        self.obs = np.concatenate(
            (
                visibility_obs,
                health_obs,
                relative_positions_obs,
                relative_distances_obs,
                unit_bits_obs,
            ),
            axis=2
        )

    def connect(self, agent, spawn=None):
        self.health[agent.id] = agent.state.max_health
        agent.state._health = self.health[agent.id:(agent.id+1)]
        self.alive[agent.id] = agent.is_alive()

        self.positions[agent.id] = spawn
        agent.state.pos = self.positions[agent.id]

        # Static data
        self.ranges[agent.id] = agent.sight_range
        self.max_health[agent.id] = agent.state.max_health
        self.unit_bits_obs[agent.id] = agent.unit_type_bits
        team_mates = [mate.id for mate in self.agents if mate.tid == agent.tid]
        self.heal_target_mask[agent.id][team_mates] = agent.has_heal()
        enemies = [enemy.id for enemy in self.agents if enemy.tid != agent.tid]
        self.attack_target_mask[agent.id][enemies] = True
        self.team_mask[agent.id] = agent.tid

    def _update_alive_status(self):
        self.alive = self.health > 0

    def calculate_avail_movements_actions(self):
        self.avail_movement_actions[:, :] = 0  # Reset
        if self.bounds is not None:
            w_steps = self.positions - [self.grid_size, 0]
            e_steps = self.positions + [self.grid_size, 0]
            n_steps = self.positions - [0, self.grid_size]
            s_steps = self.positions + [0, self.grid_size]
            stepped_positions = np.concatenate((w_steps, e_steps, n_steps, s_steps), axis=1) \
                .reshape((self.agents_n, self.get_movement_dims, 2))

            illegal_step_mask = np.ones((self.agents_n, self.get_movement_dims), dtype=np.bool)
            for pos in self.positions:
                mask = np.all(stepped_positions == pos, axis=2)
                illegal_step_mask[mask] = False

            # In bounds checks
            x_in_left_bound = stepped_positions[:, :, 0] >= 0
            x_in_right_bound = stepped_positions[:, :, 0] <= self.bounds[0]
            y_in_up_bound = stepped_positions[:, :, 1] >= 0
            y_in_down_bound = stepped_positions[:, :, 1] <= self.bounds[1]
            all_in_bound = (x_in_left_bound & x_in_right_bound) & (y_in_up_bound & y_in_down_bound)
            mask = illegal_step_mask & all_in_bound

            self.avail_movement_actions[mask] = 1.0
        else:  # unbounded map -> always add all_in_bound movement directions
            self.avail_movement_actions[:, :] = 1

    def calculate_avail_target_actions(self):
        self.avail_target_actions[:, :] = 0.0  # Reset
        mask = (self.visibility == 1) & self.alive & self.self_target_mask & (
                self.attack_target_mask | self.heal_target_mask)
        self.avail_target_actions[mask] = 1.0
