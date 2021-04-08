import logging

import gym
import numpy as np
from gym import spaces

from multiagent.core import World, Team
from multiagent.exceptions.environment_exceptions import MissingActions


class MAEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world: World,
                 reset_callback=None, reward_callback=None, observation_callback=None,
                 info_callback=None, done_callback=None,
                 global_reward=True,
                 log=False, log_level=logging.ERROR,
                 fps=None, infos=True, draw_grid=True,
                 record=False, headless=False, stream_key=None, seed=None):
        """
        Multi-Agent extension of gym.Env.

        Parameters (defaults see: in __init__)
        ----------
        @param world: World
            world object representing the whole state. Computes state transitions and internal calculations not
            relevant to the environment.

        @param reset_callback: func, optional
            provided callback to reset the environment for the next episode.
            For more info see: BaseTeamScenario in multiagent/scenarios/team/teams.py

        @param reward_callback: func, optional
            provided callback to supply agents with their reward.
            For more info see: BaseTeamScenario in multiagent/scenarios/team/teams.py

        @param observation_callback: func, optional
            provided callback to supply agents with their observation.
            For more info see: BaseTeamScenario in multiagent/scenarios/team/teams.py

        @param info_callback: func, optional
            provided callback to return additional data.
            For more info see: BaseTeamScenario in multiagent/scenarios/team/teams.py

        @param done_callback: func, optional
            provided callback to return terminal boolean.
            For more info see: BaseTeamScenario in multiagent/scenarios/team/teams.py

        @param log: bool, optional
            whether environment internals should be logged into env.log.
            This can significantly reduce performance if set to true and is only advised to debug.
            All string formatings including numpy arrays drain compute power.

        @param log_level: bool, optional
            log-level of logged environment internals.

        @param fps: int, optional
            frames-per-second of recorded video.

        @param infos: bool, optional
            whether to display additional info in the environment rendering.

        @param draw_grid: bool, optional
            whether to display the underlying movement grid of agents.

        @param record: bool, optional
            whether to display the underlying movement grid of agents.

        @param headless: bool, optional
            whether to display the underlying movement grid of agents.

        @param stream_key: str, optional
            provided twitch stream key to stream environment rendering to twitch.tv.
            If set streaming starts automatically.
        """
        self.seed = seed
        self.logger = logging.getLogger("ma-env")
        self.logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.setLevel(log_level)
        self.log = log
        if self.log:
            logging.basicConfig(filename='env.log', level=log_level)

        self.world = world
        self.agents = self.world.policy_agents
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
        self.global_reward = global_reward
        self.t = 0
        self.episode = 0
        self.episode_limit = 60
        # configure spaces
        self.action_space = []
        self.observation_space = []

        for agent in self.agents:
            # physical action space
            self.action_space.append(spaces.Discrete(self._get_action_dim(agent)))

            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32))

        self.state_n = self._get_state_dim()
        self._state = np.zeros((self.state_n,))

        # rendering
        self.headless = headless
        self.viewer = None
        if not headless:  # import rendering only if we need it (and don't import for headless machines)
            from multiagent.viewers import pygame_viewer
            self.viewer = pygame_viewer.PyGameViewer(self, fps=fps, infos=infos, draw_grid=draw_grid, record=record,
                                                     stream_key=stream_key,
                                                     headless=headless)
        self._reset_render()

    def get_env_info(self):
        return {
            "state_shape": self.state_n,
            "obs_shape": self.observation_space[0].shape[0],
            "n_actions": self.action_space[0].n,
            "n_agents": self.n,
            "episode_limit": self.episode_limit
        }

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
        :return: array of available action ids
        """
        avail_actions = [0]
        offset = 1
        avail_movement_action_indices = np.where(self.world.avail_movement_actions[agent.id])[0]
        avail_movement_action_indices += offset  # Apply offset from no-op
        avail_actions += avail_movement_action_indices.tolist()
        offset += 4  # Add movement actions to offset
        avail_target_action_indices = np.where(self.world.avail_target_actions[agent.id])[0]
        avail_target_action_indices += offset  # Apply offset from no-op + movement
        avail_actions += avail_target_action_indices.tolist()
        self.logger.debug(
            f"Agent {agent.id,} has available actions with indices: {avail_actions if self.log else None}")
        return avail_actions

    def _get_state_dim(self):
        state_dim = 0
        for a in self.world.agents:
            state_dim += len(a.self_observation) + self.world.dim_p
        return state_dim

    def _get_action_dim(self, agent):
        """
        Define how many actions can be taken by a given agent in a a given world.
        The action is constructed like the following:
        0 = No-op
        1 = Go west
        2 = Go east
        3 = Go north
        4 = Go south
        5 until 5 + # of agents = targets
        - Healers cannot attack enemies
        - Team mates cannot be attacked
        :param agent: considered agent
        :return: action space available for this agent
        """
        movement_dims = self.world.dim_p * 2  # two options for every movement axis
        attack_dims = 0
        heal_dims = 0
        # get amount of attack-able and heal-able agents in other/own team(s)
        for team in self.world.teams:
            if team.tid == agent.tid and agent.has_heal():
                heal_dims += team.size  # cannot heal himself
            if team.tid != agent.tid and not agent.has_heal():
                attack_dims += team.size
        return movement_dims + self.n + 1  # no-op
        # return movement_dims + attack_dims + heal_dims + 1  # no-op

    def step(self, action_n, heuristic_opponent=False):
        """
        Perform multi-agent environment step
        @param action_n: List of actions to take for each agent
        @param heuristic_opponent:
        """
        self.t += 1
        self.logger.info("--- Step {0}".format(self.t))
        self.logger.debug(f"Perform Actions: {action_n if self.log else None}")
        # Only consider policy agents when calling self.agents
        self.agents = self.world.policy_agents
        # Set action for each agent - this needs to be performed before stepping world !

        if len(self.agents) != len(action_n):  # Make sure we received an action for every agent
            raise MissingActions()

        for aid, agent in enumerate(self.agents):
            self._set_action(action_n[aid], agent, self.action_space[aid])

        self.logger.debug("Advance world state...")
        # Advance world state - this also sets actions in the scripted agents
        self.world.step()

        # Record observation and reward for each agent - this needs to happen after stepping world !
        # 2-d array holding all rewards of a policy agents team-wise
        team_rewards = []
        # 2-d array holding all policy agents obs
        obs_n = []
        # 1-d array holding all termination (goal) booleans for policy teams
        done_n = []
        # Extra info which does not fit into gym interface f.e. who won -> not included in done bool
        info_n = {"battle_won": [], "draw": False}

        # Go over all policy agents team-wise
        for team in self.world.policy_teams:
            # 1-d array holding all rewards of team members and as special case the win/goal reward
            local_rewards = []
            for agent in team.members:
                obs_n.append(self._get_obs(agent))
                local_rewards.append(self._get_reward(agent))

            # Check if the policy team won and add reward
            won = self._get_done(team)
            if won:
                # This is why we are not using np.mean in global reward since it would cause division by (n+1)
                local_rewards.append(200)
            done_n.append(won)

            # Calculate the reward depending on the reward function category
            if self.global_reward:
                global_reward = np.sum(local_rewards) / team.size
                team_rewards.append(global_reward)  # float
            else:
                team_rewards.append(local_rewards)  # list of floats

        for team in self.world.scripted_teams:
            # Check if scripted agents won - these agents are part of the env and do not receive obs and rewards
            done_n.append(self._get_done(team))

        info_n["battle_won"] = done_n  # Provide additional info who won the episode.

        self.logger.debug(f"Observations: {obs_n if self.log else None}")

        if self.global_reward:
            reward_n = team_rewards
            pass
            self.logger.debug(f"Global Rewards per policy controlled team: {team_rewards if self.log else None}")
        else:
            reward_n = np.concatenate(team_rewards)
            pass
            self.logger.debug(f"Local Rewards per policy controlled team: {team_rewards if self.log else None}")

        winner_id = np.where(done_n)[0]
        if len(winner_id) == 1:
            self.logger.info("------ Episode {} done - Team with id {} won the battle.".format(self.episode, winner_id))
            self.episode += 1

        # All teams won (in case of draw = both teams wiped in same step)
        elif len(winner_id) == len(self.world.teams) and self.t != self.episode_limit:
            info_n["draw"] = True

        # Episode limit reached - Place this code block after winner check !
        if self.episode_limit is not None and self.episode_limit == self.t:
            info_n["draw"] = True
            self.logger.info("------ Episode {} done - Step limit reached.".format(self.episode))
            self.episode += 1
            done_n = [True] * len(done_n)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        """
        Reset environment
        :return:
        """
        self.t = 0
        self.reset_callback(self.world)
        self._reset_render()
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
        state = []
        # State includes ALL agents. Even scripted ones !
        for agent in self.world.agents:
            agent_obs = (agent.state.pos - self.world.center) / self.world.bounds
            state = np.concatenate((state, agent_obs, agent.self_observation))
        self.logger.debug(f"State: {state if self.log else None}")
        # TOD: test instead of assertion in running code
        # assert self.state_n == len(state), "State not matching underlying dimension."
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

    def _get_done(self, team: Team):
        """
        Get terminal boolean for a particular team.
        :param team:
        :return:
        """
        return self.done_callback(team, self.world)

    def _get_reward(self, agent):
        """
        Get local reward for a particular agent.
        This reward should evaluate only the provided agents action, not the team performance.
        :param agent:
        :return:
        """
        if self.reward_callback is None:
            return 0.0
        reward = self.reward_callback(agent, self.world)
        agent.stats.reset()  # reset agent stats which were used to calculate step reward for next step
        return reward

    def _set_action(self, action, agent, time=None):
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

        # physical action index
        act_ind = action

        agent.action.u[2] = -1  # index 2(=attack) => default no attack(= -1)

        if act_ind == 0:
            return  # no-op
        elif act_ind == 1:  # WEST
            agent.action.u[0] = -1.0  # x-axis left == 1 --> index 0
        elif act_ind == 2:  # EAST
            agent.action.u[0] = +1.0  # x-axis right == 2 --> index 0
        elif act_ind == 3:  # NORTH
            agent.action.u[1] = +1.0  # y-axis up == 3 --> index 1 - pygame y axis is flipped !
        elif act_ind == 4:  # SOUTH
            agent.action.u[1] = -1.0  # y-axis down == 4 --> index 1 - pygame y axis is flipped !
        else:
            target_id = act_ind - 5  # convert action index into target id
            agent.action.u[2] = target_id  # attack >= 5 --> index 2

        # first two dimensions hold x and y axis movement -> multiply with movement step amount
        agent.action.u[:2] *= self.world.grid_size

    def _reset_render(self):
        """
        Reset rendering assets
        :return:
        """
        if self.viewer is not None:
            self.viewer.entities = None
            self.viewer.clear()

    def close(self):
        if self.viewer:
            self.viewer.close()
        del self.world
        self.world = None

    def render(self, mode='human'):
        """
        Render the environment via the defined PyGameViewer.
        @param mode:
        @param headless:
        @return:
        """

        if not self.headless:
            if self.viewer is None:
                # import rendering only if we need it (and don't import for headless machines)
                from multiagent.viewers import pygame_viewer
                self.viewer = pygame_viewer.PyGameViewer(self, headless=False)
            # create rendered entities
            if self.viewer.entities is None:
                # import rendering only if we need it (and don't import for headless machines)
                from multiagent.viewers import pygame_viewer
                self.viewer.init(self.world.entities)

            #
            # main render loop part
            #
            self.viewer.clear()

            self.viewer.update()

            self.viewer.render()
        else:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        return None


class TeamsEnv(MAEnv):
    def __init__(self, match_build_plan, grid_size, **kwargs):
        from multiagent.scenarios import team
        scenario = team.load("teams.py").TeamsScenario(match_build_plan)
        world = scenario.make_teams_world(grid_size)
        super().__init__(world,
                         reset_callback=scenario.reset_world,
                         reward_callback=scenario.reward,
                         observation_callback=scenario.observation,
                         done_callback=scenario.done, **kwargs)


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
