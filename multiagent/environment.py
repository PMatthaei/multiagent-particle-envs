import logging
import sys

import gym
import numpy as np
from gym import spaces

from multiagent.core import World


class MAEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world: World, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, log=False, log_level=logging.DEBUG):
        if log:
            logging.basicConfig(filename='env.log', level=log_level)
            self.logger = logging.getLogger("ma-env")
            self.logger.handlers = []
            ch = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.setLevel(log_level)

        self.world = world
        self.agents = self.world.policy_agents
        self.movement_step_amount = self.world.grid_size  # move X pixels per action - de-facto grid size
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        # if true, every agent has the same reward
        self.global_reward = world.collaborative
        self.t = 0
        self.episode = 0
        self.episode_limit = 60
        # configure spaces
        self.state_n = self.n * 4  # Four features per agent
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            # physical action space
            self.action_space.append(spaces.Discrete(self._get_action_dim(agent)))

            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.viewer = None
        self._reset_render()

    def get_env_info(self):
        return {"state_shape": self.state_n,
                "obs_shape": self.observation_space[0].shape[0],
                "n_actions": self.action_space[0].n,
                "n_agents": self.n,
                "episode_limit": self.episode_limit}

    def get_available_actions(self, agent):
        """
        Vector form of @get_available_action_ids
        @param agent:
        @return:
        """
        avail_actions = np.zeros((self._get_action_dim(agent),))
        ids = self.get_available_action_ids(agent)
        avail_actions[ids] = 1
        return avail_actions

    def get_available_action_ids(self, agent):
        """
        Calculate possible actions for an agent. These actions can be restricted by available agents to attack or
        move-able directions if near a world bound. This functions currently does NOT provide available agents by
        sight range. This means that a agent might be listed in the available actions array but in the world is not
        accessible due being out of reach.
        :param agent:
        :return: array of available action indices
        """
        avail_actions = []
        if agent.is_dead():  # when dead only allow no-op action
            return [0]
        if self.world.bounds is not None:
            x = agent.state.pos[0]
            y = agent.state.pos[1]
            if x - self.movement_step_amount >= 0:  # WEST would not exceed bounds
                avail_actions.append(1)
            if x + self.movement_step_amount <= self.world.bounds[0]:  # EAST would not exceed bounds
                avail_actions.append(2)
            if y - self.movement_step_amount >= 0:  # NORTH would not exceed bounds
                avail_actions.append(3)
            if y + self.movement_step_amount <= self.world.bounds[1]:  # SOUTH would not exceed bounds
                avail_actions.append(4)
        else:  # unbounded map -> always add all movement directions
            avail_actions.append([1, 2, 3, 4])

        act_ind_offset = 5  # Attack/Heal actions begin at index 5

        # All alive agents (except self) can be taken a action against f.e heal, attack etc
        avail_actions = avail_actions + [ag.id + act_ind_offset for ag in self.world.alive_agents if
                                         # Include the following agents ids as action encoded ids
                                         # NOT the agent itself -> TODO: Remove if self-heal needed
                                         agent.id != ag.id and
                                         # enemy ids if the agent is not a healer
                                         ((agent.tid != ag.tid and not agent.has_heal()) or
                                          # team mates ids if the agent is a healer
                                          (agent.has_heal() and agent.tid == ag.tid))]

        self.logger.debug("Agent {} has available actions with ids: {}".format(agent.id, avail_actions))
        return avail_actions

    def _get_action_dim(self, agent):
        """
        Define how many actions can be taken by a given agent in a a given world.
        The action is constructed like the following:
        0 = No-op
        1 = Go west
        2 = Go east
        3 = Go north
        4 = Go south
        5 until 5 + #a (= of attackable enemies) = attack agent with id x
        5 + #a until (5 + #a) + #h = heal agent with id y
        :param agent: considered agent
        :return: action space available for this agent
        """
        movement_dims = self.world.dim_p * 2
        attack_dims = 0
        heal_dims = 0
        # get amount of attack-able and heal-able agents in other/own team(s)
        for team in self.world.teams:
            if team.tid == agent.tid and agent.has_heal():
                heal_dims += len(team.members)  # cannot heal himself
            if team.tid != agent.tid and not agent.has_heal():
                attack_dims += len(team.members)
        return movement_dims + len(self.world.agents) + 1  # no-op
        # TODO: Current impl assumes same action space for all agents but some actions will always be unavailable:
        #   - Healers cannot attack enemies
        #   - Team mates cannot be attacked
        # return movement_dims + attack_dims + heal_dims + 1  # no-op

    def step(self, action_n):
        """
        Perform multi-agent environment step
        :param action_n:
        :return:
        """
        self.t += 1
        self.logger.debug("--- Step {0}".format(self.t))
        self.logger.debug("Actions: {0}".format(action_n))

        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent - this needs to be performed before stepping world !
        for aid, agent in enumerate(self.agents):
            self._set_action(action_n[aid], agent, self.action_space[aid])
        # advance world state
        self.world.step()
        # record observation for each agent - this needs to happen after stepping world !
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        self.logger.debug("Obs: {0}".format(obs_n))

        if self.global_reward:
            # Implementation see: On local rewards and scaling distributed reinforcement learning
            global_reward = np.mean(reward_n)
            reward_n = [global_reward] * self.n

        self.logger.debug("Rewards: {0}".format(reward_n))

        if any(done_n):
            self.logger.debug("------ Done: {0}".format(done_n))

        if self.episode_limit == self.t:
            done_n = [True] * len(done_n)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        """
        Reset environment
        :return:
        """
        self.t = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    def _get_info(self, agent):
        """
        Get info used for benchmarking
        :param agent:
        :return:
        """
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def get_state(self):
        """Returns the global state.
        NOTE: This function should not be used during decentralised execution.
        """
        cx, cy = self.world.center
        state = np.array([])
        for agent in self.agents:
            x = (agent.state.pos[0] - cx) / self.world.bounds[0]  # relative X
            y = (agent.state.pos[1] - cy) / self.world.bounds[1]  # relative Y
            agent_state = np.array([[x, y], agent.self_observation]).flatten()
            state = np.append(state, agent_state)
        self.logger.debug("State: {0}".format(state))
        return state

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return [self._get_obs(agent) for agent in self.agents]

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = [self.get_available_actions(agent) for agent in self.agents]
        return avail_actions

    def _get_obs(self, agent):
        """
        Get observation for a particular agent
        :param agent:
        :return:
        """
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    def _get_done(self, agent):
        """
        Get dones for a particular agent. A environment is done if it reached its max steps
        :param agent:
        :return:
        """
        if self.episode_limit is not None and self.episode_limit == self.t:
            self.episode += 1
            self.logger.debug("------ Episode: {0}".format(self.episode))
            return True
        if self.done_callback is None:
            return False
        if self.done_callback(agent, self.world):
            self.episode += 1
            self.logger.debug("------ Episode: {0}".format(self.episode))
        return self.done_callback(agent, self.world)

    def _get_reward(self, agent):
        """
        Get local reward for a particular agent.
        This reward should evaluate only the provided agents action, not the team performance.
        :param agent:
        :return:
        """
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def _set_action(self, action, agent, action_space, time=None):
        """
        Set env action for a particular agent. Convert action index to environment/world state update and
        provide these state changes to the agent for later update f.e of position and stats.
        :param action: the performed action by
        :param agent: the corresponding agent
        :param action_space: the action space of the agent
        :param time: the current time step
        :return: None
        """
        # set default actions (physical and communication)
        agent.action.u = np.zeros(self.world.dim_p + 1)
        agent.action.c = np.zeros(self.world.dim_c)

        # physical action index
        act_ind = action

        agent.action.u[2] = -1  # index 2(=attack) => default no attack(= -1)

        if act_ind == 0:
            return  # no-op
        elif act_ind == 1:
            agent.action.u[0] = -1.0  # x-axis left == 1 --> index 0
        elif act_ind == 2:
            agent.action.u[0] = +1.0  # x-axis right == 2 --> index 0
        elif act_ind == 3:
            agent.action.u[1] = -1.0  # y-axis up == 3 --> index 1
        elif act_ind == 4:
            agent.action.u[1] = +1.0  # y-axis down == 4 --> index 1
        else:
            target_id = act_ind - 5  # convert action index into target id
            agent.action.u[2] = target_id  # attack >= 5 --> index 2

        # first two dimensions hold x and y axis movement -> multiply with movement step amount
        agent.action.u[:2] *= self.movement_step_amount

    def _reset_render(self):
        """
        Reset rendering assets
        :return:
        """
        if self.viewer is not None:
            self.viewer.entities = None
            self.viewer.clear()

    def render(self, mode='human'):
        """
        Render the environment via the defined PyGameViewer.
        :param mode:
        :return:
        """
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            if message is not None and not message and message != '':
                print("Communicated message: ", message)

        if self.viewer is None:
            # import rendering only if we need it (and don't import for headless machines)
            from multiagent import pygame_rendering
            self.viewer = pygame_rendering.PyGameViewer(self)

        # create rendered entities
        if self.viewer.entities is None:
            # import rendering only if we need it (and don't import for headless machines)
            from multiagent import pygame_rendering
            self.viewer.init(self.world.entities)

        #
        # main render loop part
        #
        self.viewer.clear()

        self.viewer.update()

        self.viewer.render()
        return None


class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        """
        Vectorized wrapper for a batch of multi-agent environments.
        Assumes all environments have the same observation and action space.
        :param env_batch:
        """
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.clear()
        return obs_n

    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
