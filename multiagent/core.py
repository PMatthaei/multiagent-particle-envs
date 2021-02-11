import math
from enum import Enum, IntEnum

import numpy as np

from multiagent.exceptions.agent_exceptions import NoTargetFoundError, IllegalTargetError
from multiagent.exceptions.world_exceptions import NoTeamFoundError
import logging


class RoleTypes(Enum):
    TANK = {"max_health": 100, "attack_damage": 5}
    ADC = {"max_health": 50, "attack_damage": 20}
    HEALER = {"max_health": 75, "attack_damage": 20, "can_heal": True}


class UnitAttackTypes(Enum):
    RANGED = {"attack_range": 35}
    MELEE = {"attack_range": 5}


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
        self.pos = spawn_pos


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
        self.bounding_circle_radius = 6
        self.name = ''
        # entity can move / be pushed
        self.movable = False
        # color
        self.color = [0.75, 0, 0]

        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    def can_see(self, other):
        dist = np.linalg.norm(self.state.pos - other.state.pos)
        return dist <= self.sight_range

    def is_alive(self):
        return self.state.health >= 0

    def is_dead(self):
        return not self.is_alive()

    @property
    def mass(self):
        return self.initial_mass


# properties of world objects
class WorldObject(Entity):
    def __init__(self):
        super(WorldObject, self).__init__()


class Team:
    def __init__(self, tid, members):
        """
        :param tid: Identifier for this team
        :param members: Members in this team
        """
        self.tid = tid
        self.members = members

    def is_wiped(self):
        return all([agent.is_dead() for agent in self.members])


class PerformanceStatistics:
    def __init__(self, kills=0, assists=0, dmg_dealt=0, dmg_healed=0, attacks_performed=0, heals_performed=0,
                 distance_traveled=0):
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
        self.dmg_dealt = dmg_dealt
        self.dmg_healed = dmg_healed
        self.attacks_performed = attacks_performed
        self.heals_performed = heals_performed
        self.distance_traveled = distance_traveled

        self.prev_t_stats: PerformanceStatistics = None

    @property
    def delta(self):
        """
        :return: Agents stats corresponding to the last registered time step.
        """
        return PerformanceStatistics(
            kills=self.kills - self.prev_t_stats.kills,
            assists=self.assists - self.prev_t_stats.assists,
            dmg_dealt=self.dmg_dealt - self.prev_t_stats.dmg_dealt,
            dmg_healed=self.dmg_healed - self.prev_t_stats.dmg_healed,
            attacks_performed=self.attacks_performed - self.prev_t_stats.attacks_performed,
            heals_performed=self.heals_performed - self.prev_t_stats.heals_performed,
            distance_traveled=self.distance_traveled - self.prev_t_stats.distance_traveled,
        )


# properties of agent entities
class Agent(Entity):
    def __init__(self, id, tid, color, build_plan):
        super(Agent, self).__init__()
        self.id = id
        # team id
        self.tid = tid
        self.name = 'Agent %d' % id
        self.color = color
        self.attack_type = build_plan['attack_type'].value
        self.role = build_plan['role'].value

        self.attack_range = self.attack_type['attack_range']
        # sight range cannot be smaller than attack range
        self.sight_range = max(self.attack_range, self.sight_range)
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
        self.action_callback = None

    @property
    def self_observation(self):
        return [
            self.state.health / self.state.max_health,  # relative health
            self.state.shield / self.state.max_shield if self.state.max_shield != 0 else 0.0  # relative shield
        ]

    def observe(self, other):
        """
        Observe another agent.
        @param other: Agent which is observed by this agent
        @return: The observation made of the provided agent
        """
        if self.can_see(other):
            rel_pos = self.state.pos - other.state.pos
            return [
                rel_pos[0] / self.sight_range,  # x position relative to observer
                rel_pos[1] / self.sight_range,  # y position relative to observer
                self.state.health / self.state.max_health,  # relative health
                self.state.shield / self.state.max_shield if self.state.max_shield != 0 else 0.0  # relative shield
            ]
        else:
            return [0] * 4

    def can_attack(self, other: Entity):
        dist = np.linalg.norm(self.state.pos - other.state.pos)
        return dist <= self.attack_range and other.is_alive()

    def heal(self, other):
        other.state.health += self.attack_damage
        self.stats.dmg_healed += self.attack_damage
        logging.debug("Agent {0} in team {1} healed Agent {2} in team {3} for {4}"
                      .format(self.id, self.tid, other.id, other.tid, self.attack_damage))

    def attack(self, other):
        other.state.health -= self.attack_damage
        logging.debug("Agent {0} in team {1} attacked Agent {2} in team {3} for {4}"
                      .format(self.id, self.tid, other.id, other.tid, self.attack_damage))
        self.stats.dmg_dealt += self.attack_damage
        if other.is_dead():
            self.stats.kills += 1
            logging.debug("Agent {0} is dead.".format(other.id))

    def _has_heal(self):
        return 'can_heal' in self.role and self.role['can_heal']

    def can_heal(self, target=None):
        """
        Test if a unit can heal in general. If target provided test for healing that specific agent.
        :param target:
        :return:
        """
        return self._has_heal() and (target is not None and target.tid != self.tid) and target.is_alive()


class World(object):
    def __init__(self, grid_size, bounds=(1280, 720)):
        """
        Multi-agent world
        :param bounds: World bounds in which the agents can move
        """
        self.bounds = bounds
        self.grid_size = grid_size
        # indicates if the reward will be global(cooperative) or local
        self.collaborative = False
        # list of teams build by a subset of ...
        self.teams = []
        # list of agents
        self.agents = []
        # list of non-agent objects in the world
        self.objects = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3

    def get_team_members(self, agent):
        return [member for member in self.get_team(agent.tid).members if member.id != agent.id]

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

    def get_opposing_teams(self, tid):
        """
        Return opposing teams of the team with the provided team id
        :param tid: Identifier for a team
        :return: Teams NOT belonging to the identifier
        """
        return [team for team in self.teams if team.tid != tid]

    def get_visible_entities(self, entity: Entity):
        """
        Return all entities visible to the provided entity. If no range given or zero return empty list.
        @param entity: The entity from which to apply a range query onto its neighbor entities.
        @return: List of visible entities for the provided entity
        """
        if entity.sight_range is None or entity.sight_range == 0:
            return []
        return [e for e in self.entities if entity.can_see(e)]

    def get_available_movement(self, agent):
        if self.bounds is not None:
            avail_movement = [0] * 4  # four movement dims
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
            return [1] * 4  # four movement dims

    @property
    def alive_agents(self):
        return [agent for agent in self.agents if agent.is_alive()]

    @property
    def grid_center(self):
        center_x = self.bounds[0] / 2.0
        center_y = self.bounds[1] / 2.0
        return center_x - (center_x % self.grid_size), center_y - (center_y % self.grid_size)

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

    def step(self):
        """
        Update state of the world
        :return:
        """
        # Set actions for scripted/heuristic agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # Update each alive agent - agent action was set during environment.py step() function
        # Shuffle randomly to prevent favoring
        import random
        agents = self.alive_agents
        random.shuffle(agents)
        for agent in agents:

            # Update position
            move_vector = agent.action.u[:2]
            agent.state.pos += move_vector
            if any([dim > 0 for dim in move_vector]):  # is there movement greater zero?
                logging.debug("Agent {0} moved by {1}:".format(agent.id, move_vector))

            # Influence entity if target set
            agent_has_action_target = agent.action.u[2] != -1
            if agent_has_action_target:
                agent.target_id = int(agent.action.u[2])
                if agent.target_id is None or math.isnan(agent.target_id):
                    raise NoTargetFoundError()
                target = self.agents[agent.target_id]
                if agent.can_heal(target):
                    agent.heal(target)
                elif agent.can_attack(target):
                    agent.attack(target)
                    if target.is_dead():  # Target died due to the attack
                        # TODO what to do if more than one unit attacks the target in the same step and it dies by one of them?
                        pass
                else:
                    # TODO: For now, illegal actions can be taken and are available but will not change environment
                    logging.warning("Agent {0} cannot attack Agent {1} due to range.".format(agent.id, agent.target_id))
