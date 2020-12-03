import math
from enum import Enum, IntEnum

import numpy as np

from multiagent.exceptions.agent_exceptions import NoTargetFoundError, IllegalTargetError
from multiagent.exceptions.world_exceptions import NoTeamFoundError
import logging


class SkillTypes(Enum):
    HEAL = 0


class UnitAttackTypes(Enum):
    RANGED = 0,
    MELEE = 1


class ActionTypes(IntEnum):
    NOOP = 0,
    WEST = 1,
    EAST = 2,
    SOUTH = 3,
    NORTH = 4,


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.max_health = 0
        self.max_shield = 0
        self.health = 0
        self.shield = 0

    def reset(self, spawn_pos):
        self.health = self.max_health
        self.shield = self.max_shield
        self.p_pos = spawn_pos


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
        self.sight_range = 20
        # radius defines entity`s collision and visuals
        self.bounding_circle_radius = 4
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    def can_see(self, other):
        dist = np.linalg.norm(self.state.p_pos - other.state.p_pos)
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
        :param tid:
        :param members:
        """
        self.tid = tid
        self.members = members
        # Set ids in agents
        [self.assign(agent) for agent in self.members]

    def assign(self, agent):
        agent.tid = self.tid


class AgentStats:
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

        self.prev_t_stats: AgentStats = None

    @property
    def delta(self):
        """
        :return: Agents stats corresponding to the last registered time step.
        """
        return AgentStats(
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
    def __init__(self):
        super(Agent, self).__init__()
        # team id
        self.tid = None
        # agents are movable by default
        self.capabilities = []
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # stats
        self.stats = AgentStats()
        # script behavior to execute
        self.action_callback = None

    def can_attack(self, other: Entity):
        dist = np.linalg.norm(self.state.p_pos - other.state.p_pos)
        return dist <= self.attack_range and other.is_alive()

    def heal(self, other):
        other.state.health += self.attack_damage
        logging.debug("Agent {0} in team {1} healed Agent {2} in team {3} for {4}"
                      .format(self.id, self.tid, other.id, other.tid, self.attack_damage))

    def attack(self, other):
        other.state.health -= self.attack_damage
        logging.debug("Agent {0} in team {1} attacked Agent {2} in team {3} for {4}"
                      .format(self.id, self.tid, other.id, other.tid, self.attack_damage))
        if other.is_dead():
            logging.debug("Agent {0} is dead.".format(other.id))

    def can_heal(self, target=None):
        """
        Test if a unit can heal in general. If target provided test for healing that specific agent.
        :param target:
        :return:
        """
        return SkillTypes.HEAL in self.capabilities and (target is None or target.tid == self.tid) and target.is_alive()


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

    def get_team(self, tid):
        """
        Return the team with the given id
        :param tid:
        :return:
        """
        for team in self.teams:
            if team.tid == tid:
                return team
        raise NoTeamFoundError(tid)

    def get_opposing_teams(self, tid):
        """
        Return opposing teams of the team with the provided team id
        :param tid:
        :return:
        """
        return [team for team in self.teams if team.tid != tid]

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
            # Set communication state directly
            if agent.silent:
                agent.state.c = np.zeros(self.dim_c)
            else:
                noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
                agent.state.c = agent.action.c + noise

            # Update position
            move_by = agent.action.u[:2]
            agent.state.p_pos += move_by
            logging.debug("Agent {0} moved by {1}:".format(agent.id, move_by))

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
                        pass  # TODO what to do if more than one unit attacks the target and it dies by one of them?
                else:
                    # TODO: For now, illegal actions can be taken and are available but will not change environment
                    logging.warning("Agent {0} cannot attack Agent {1} due to range.".format(agent.id, agent.target_id))
