import numpy as np

from maenv.core import World, Action, Agent
from maenv.interfaces.ai import ScriptedAI


class BasicScriptedAI(ScriptedAI):

    def act(self, agent: Agent, world: World) -> Action:
        """
        Defines how a given agent should act in a given world.
        In this case the agent always targets the closest agent which is targetable.
        If no agent is targetable (out of sight) the AI moves in direction of the closest.
        @param agent:
        @param world:
        @return: action
        """
        action = Action()
        action.u = np.zeros(world.dim_p + 1)
        action.u[2] = -1  # default is no target == -1

        # TODO: Vectorize
        if world.distances is not None:  # if distance matrix initialized
            masked_distances = world.distances[agent.id].copy()
            if agent.has_heal():
                mask = (world.team_affiliations != agent.tid) | (world.alive == 0)  # mask out all enemies and dead
            else:
                mask = (world.team_affiliations == agent.tid) | (world.alive == 0)  # mask out all teammates and dead
            masked_distances[agent.id] = np.inf  # infinite distance to self to prevent to be chosen as target
            masked_distances[mask] = np.inf  # infinite distance all non-targetable agents
            target_id = np.argmin(masked_distances)
            distance = masked_distances[target_id]

            if distance <= agent.sight_range:  # set closest agent as target if in range
                action.u[2] = target_id  # attack >= 5 --> index 2
            else:  # move towards the closest agent if not in range
                closest_agent = world.agents[target_id]
                position_difference = world.positions[closest_agent.id] - world.positions[agent.id]
                max_difference_dimension = np.argmax(np.abs(position_difference))
                max_diff = position_difference[max_difference_dimension]
                action.u[max_difference_dimension] = np.sign(max_diff)
        else:  # No-Op
            action.u[0] = 0
            action.u[1] = 0

        # first two dimensions hold x and y axis movement -> multiply with movement step amount
        action.u[:2] *= world.grid_size  # (=movement step size)
        return action
