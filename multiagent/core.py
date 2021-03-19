from __future__ import annotations

import itertools
import logging
import math
from enum import Enum, IntEnum

import numpy as np

from multiagent.exceptions.agent_exceptions import NoTargetFoundError, IllegalTargetError
from multiagent.exceptions.world_exceptions import NoTeamFoundError

logger = logging.getLogger("ma-env")


class RoleTypes(Enum):
    TANK = {"max_health": 100, "attack_damage": 5}
    ADC = {"max_health": 50, "attack_damage": 50}
    HEALER = {"max_health": 75, "attack_damage": 20, "can_heal": True}


class UnitAttackTypes(Enum):
    RANGED = {"attack_range": 35}
    MELEE = {"attack_range": 5}


# Calculate all unique unit types
UNIQUE_UNIT_TYPES = list(itertools.product(RoleTypes, UnitAttackTypes))
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
        self.max_health = 100
        self.max_shield = 0
        self.health = 0
        self.shield = 0

    def reset(self, spawn_pos):
        self.health = self.max_health
        self.shield = self.max_shield
        self.pos = spawn_pos.copy()  # ! we do not want to modify the original spawn in-place !


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

    def reset(self, spawn_pos):
        super().reset(spawn_pos)
        self.c = None


# action of the agent
class Action(object):
    def __init__(self, index=None, owner=None, target=None, u=None, c=None):
        """
        # TODO
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
        """
        :param tid: Identifier for this team
        :param members: Members in this team
        """
        self.tid = tid
        self.members = members
        self.is_scripted = is_scripted
        self.size = len(members)  # team size not influenced by deaths

    def is_wiped(self):
        return all([agent.is_dead() for agent in self.members])


class PerformanceStatistics:
    def __init__(self, kills=0, assists=0, dmg_dealt=0, dmg_healed=0, attacks_performed=0, heals_performed=0,
                 distance_traveled=0, dmg_received=0):
        """
        Holds the stats of an agent within an episode, representing it at the current time step t.
        :param kills:
        :param assists:
        :param dmg_dealt:
        :param dmg_healed:
        :param attacks_performed:
        :param heals_performed:
        :param distance_traveled:
        """
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
        self.sight_range = self.attack_range  # TODO: sight range != attack range
        assert self.sight_range >= self.attack_range, "Sight range cannot be smaller than attack range."
        self.attack_damage = self.role['attack_damage']
        self.max_health = self.role['max_health']

        self.movable = True
        # control range
        # state
        self.state = AgentState()
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
        """
        Test if a unit can heal in general. If target provided test for healing that specific agent.
        :param target:
        :return:
        """
        return self.has_heal() and (target is not None and target.tid != self.tid) and target.is_alive()


class World(object):
    def __init__(self, grid_size: int, agents_n: int, bounds=np.array([1280, 720]), log=False):
        """
        Multi-agent world
        :param bounds: World bounds in which the agents can move
        """
        self.bounds = bounds
        self.log = log
        self.occupied_positions = None
        self.grid_size = grid_size
        # indicates if the reward will be global(cooperative) or local
        self.collaborative = False
        # list of teams build by a subset of ...
        self.teams = []
        self.teams_wiped = []
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

        self.occupied_positions = np.zeros((agents_n, self.dim_p + 1))
        self.distance_matrix = np.full((agents_n, agents_n), 0.0)  # Assumes all enemies seen -> first pruning
        self.visibility_matrix = np.full((agents_n, agents_n), False)

    def get_team(self, tid):
        """
        Return the team with the given id
        :param tid: Identifier for a team
        :return: Team belonging to the identifier
        """
        for team in self.teams:
            if team.tid == tid:
                return team
        raise NoTeamFoundError(tid)

    def get_opposing_teams(self, tid: int):
        """
        Return opposing teams of the team with the provided team id
        :param tid: Identifier for a team
        :return: Teams NOT belonging to the identifier
        """
        return [team for team in self.teams if team.tid != tid]

    def is_free(self, pos: np.array):
        alive = self.occupied_positions[:, 2]  # occupied may also hold pos of dead agents -> mask out via alive
        # the desired pos matches occupied pos in all dimensions
        pos_occupied = np.all(self.occupied_positions[:, [0, 1]] == pos, axis=1)
        # desired position is occupied AND the entity on this pos is alive
        pos_occupied = np.logical_and(pos_occupied, alive)
        # if there is not one position found the desired pos is free to move to
        return not np.any(pos_occupied)

    def get_movement_dims(self):
        return self.dim_p * 2

    def get_available_movement(self, agent: Agent):
        avail_movement = [0] * self.get_movement_dims()  # four movement dims

        if agent.is_dead():
            return avail_movement  # four movement dims

        if self.bounds is not None:
            x = agent.state.pos[0]
            y = agent.state.pos[1]
            if x - self.grid_size >= 0:  # WEST would not exceed bounds
                avail_movement[0] = 1
            if x + self.grid_size <= self.bounds[0]:  # EAST would not exceed bounds
                avail_movement[1] = 1
            if y - self.grid_size >= 0:  # NORTH would not exceed bounds
                avail_movement[2] = 1
            if y + self.grid_size <= self.bounds[1]:  # SOUTH would not exceed bounds
                avail_movement[3] = 1
            return avail_movement
        else:
            return [1] * self.get_movement_dims()  # four movement dims

    def _get_obs_dims(self, agent: Agent):
        obs_dims = self.dim_p  # position
        obs_dims += 1  # visibility bool
        obs_dims += 1  # distance
        obs_dims += 1  # health
        obs_dims += agent.unit_type_bits_n
        return obs_dims

    def get_obs(self, observer: Agent, target: Agent):
        """
        Retrieve observation conducted by an agent on another agent.
        @param observer: the agent observing his environment
        @param target: agent which is observed
        @return: the observation made of the provided agent
        """
        # TODO vectorize
        obs_target_visible = self.visibility_matrix[observer.id][target.id]
        if obs_target_visible and target.is_alive():
            rel_pos = target.state.pos - observer.state.pos  # TODO can be calced in batch
            distance = self.distance_matrix[observer.id][target.id]
            obs = [
                      obs_target_visible,  # is the observed unit visible
                      distance / observer.sight_range,  # distance relative to sight range #TODO can be calced in batch
                      rel_pos[0] / observer.sight_range,  # x position relative to observer #TODO can be calced in batch
                      rel_pos[1] / observer.sight_range,  # y position relative to observer #TODO can be calced in batch
                      target.state.health / target.state.max_health,  # relative health #TODO can be calced in batch
                  ] + target.unit_type_bits
            # TODO move to test
            # assert len(obs) == self._get_obs_dims(target), "Check observation matches underlying dimension."
            return obs
        else:
            return [0] * self._get_obs_dims(target)

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
        """
        :return: all entities in the world
        """
        return self.agents + self.objects

    @property
    def policy_teams(self):
        return [team for team in self.teams if not team.is_scripted]

    @property
    def scripted_teams(self):
        return [team for team in self.teams if team.is_scripted]

    @property
    def policy_agents(self):
        """
        :return: all agents controllable by external policies (trained AI)
        """
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        """
        :return: all agents controlled by world scripts (heuristic, non-training AI)
        """
        return [agent for agent in self.agents if agent.action_callback is not None]

    def can_attack(self, agent: Agent, target: Agent):
        if target.tid == agent.tid:
            return False
        return self.visibility_matrix[agent.id][target.id] and target.is_alive()

    def step(self):
        """
        Update state of the world.
        """
        illegal_target_actions = 0
        illegal_movement_actions = 0
        # Set actions for scripted/heuristic agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # Update each alive agent - agent action was set during environment.py step() function
        # Shuffle randomly to prevent favoring
        import random
        shuffled_agents = self.alive_agents.copy()
        random.shuffle(shuffled_agents)

        for agent in shuffled_agents:

            # Call before updating occupied positions !
            self._update_pos(agent, illegal_movement_actions)

            # Mark position of agent and its state -> only occupied if alive
            self.occupy_pos(agent)

            self._calculate_visibility(agent)

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
                    if target.is_dead():
                        # Mark dead
                        self.occupied_positions[target.id, 2] = 0.0
                else:
                    illegal_target_actions += 1
                    if self.log:
                        logger.debug(
                            "Agent {0} cannot attack Agent {1} due to range.".format(agent.id, agent.target_id))

                agent.target_id = None  # Reset target after processing

        # After update test if world is done aka only one team left
        self.teams_wiped = [team.is_wiped() for team in self.teams]

    def _update_pos(self, agent, illegal_movement_actions):
        move_vector = agent.action.u[:2]
        agent.state.pos += move_vector
        has_moved = any(move_vector)
        if has_moved and not self.is_free(agent.state.pos):
            illegal_movement_actions += 1
            agent.state.pos -= move_vector

    def occupy_pos(self, agent):
        self.occupied_positions[agent.id, :2] = agent.state.pos
        self.occupied_positions[agent.id, 2] = agent.is_alive()

    def _calculate_visibility(self, agent):
        # Calculate all distances to other agents
        all_pos = self.occupied_positions[:, :2]
        alive = self.occupied_positions[:, 2] == 1.0
        self.distance_matrix[agent.id, alive] = np.linalg.norm(all_pos[alive] - agent.state.pos, axis=1)
        self.visibility_matrix[agent.id, alive] = self.distance_matrix[agent.id, alive] <= agent.sight_range
        self.visibility_matrix[alive == 0.0] = False
