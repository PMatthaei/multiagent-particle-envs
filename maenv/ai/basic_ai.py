import numpy as np

from maenv.core import World, Action, Agent
from maenv.interfaces.ai import ScriptedAI


class BasicScriptedAI(ScriptedAI):

    def act(self, agent: Agent, world: World) -> Action:
        """
        Defines how a given agent should act in a given world.
        In this case the agent always targets the closest agent which is attackable.
        If no agent is attackable (out of sight) the AI moves in direction of the closest.
        @param agent:
        @param world:
        @return: action
        """
        self.action = super(BasicScriptedAI, self).act(agent, world)
        # TODO: Vectorize
        if world.distances is not None:  # if distance matrix initialized
            masked_distances = self._get_masked_distances(agent, world)
            target_id = self._get_target(masked_distances, world)
            distance = masked_distances[target_id]

            if distance <= agent.sight_range:  # set closest agent as target if in range
                self.action.u[2] = target_id  # attack >= 5 --> index 2
            else:  # move towards the closest agent if not in range
                closest_agent = world.agents[target_id]
                position_difference = world.positions[closest_agent.id] - world.positions[agent.id]
                max_difference_dimension = np.argmax(np.abs(position_difference))
                max_diff = position_difference[max_difference_dimension]
                self.action.u[max_difference_dimension] = np.sign(max_diff)
        else:  # No-Op
            self.action.u[0] = 0
            self.action.u[1] = 0

        # first two dimensions hold x and y axis movement -> multiply with movement step amount
        self.action.u[:2] *= world.grid_size  # (=movement step size)
        return self.action

    def _get_target(self, masked_distances, world) -> int:
        """
        Get closest agent id as target
        @param masked_distances:
        @param world:
        @return: id of the target
        """
        target_id = np.argmin(masked_distances)
        return target_id

    def _get_masked_distances(self, agent: Agent, world: World) -> np.array:
        """
        Mask distances depending on the agent role. Healers should only target team mates which alive while attacking
        agents should target enemies which are alive. Therefore the counter part should be masked out as non attackable.
        @param agent:
        @param world:
        @return:
        """
        masked_distances = world.distances[agent.id].copy()
        if agent.has_heal():  # mask out all enemies and dead
            non_target_mask = (world.team_affiliations != agent.tid) | (world.alive == 0)
        else:  # mask out all teammates and dead
            non_target_mask = (world.team_affiliations == agent.tid) | (world.alive == 0)
        masked_distances[agent.id] = np.inf  # infinite distance to self to prevent to be chosen as target
        masked_distances[non_target_mask] = np.inf  # infinite distance all non-attackable agents
        return masked_distances
