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
        action = Action()
        action.u = np.zeros(world.dim_p + 1)  # reset previous action
        action.u[2] = -1  # default is no target == -1

        masked_distances = self._get_masked_distances(agent, world)
        if np.all(np.isinf(masked_distances)):
            return action  # distances undefined -> no-op
        target_id = self._get_target(masked_distances, world)
        closest_agent = world.agents[target_id]
        distance = masked_distances[target_id]
        if distance <= (agent.sight_range * world.grid_size):  # set closest agent as target if in range
            action.u[2] = target_id  # attack >= 5 --> index 2
        else:  # move towards the closest agent if not in range
            agent_pos = world.positions[agent.id]

            position_difference = world.positions[closest_agent.id] - agent_pos
            max_difference_dimension = np.argmax(np.abs(position_difference))
            max_diff = position_difference[max_difference_dimension]
            action.u[max_difference_dimension] = np.sign(max_diff)
            # first two dimensions hold x and y axis movement -> multiply with movement step amount
            action.u[:2] *= world.grid_size  # (=movement step size)
            new_pos = agent_pos + action.u[:2]
            if not world.is_free(new_pos):  # if the stepped pos is occupied -> search free pos to move to
                possible_steps = np.expand_dims(agent_pos, axis=0).repeat(4, axis=0) + world.moves
                np.testing.assert_array_equal(world.stepable_positions[agent.id], possible_steps)
                free = np.array([world.is_free(pos) for pos in world.stepable_positions[agent.id]])
                move = world.moves[np.argmax(free > 0)]
                action.u[:2] = move
        return action

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
        masked_distances = np.copy(world.distances[agent.id])
        if agent.has_heal():  # mask out all enemies or dead
            non_target_mask = (world.team_affiliations != agent.tid) | (world.alive == 0)
        else:  # mask out all teammates or dead
            non_target_mask = (world.team_affiliations == agent.tid) | (world.alive == 0)
        masked_distances[agent.id] = np.inf  # infinite distance to self to prevent to be chosen as target
        masked_distances[non_target_mask] = np.inf  # infinite distance all non-attackable agents
        return masked_distances
