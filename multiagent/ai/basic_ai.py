import numpy as np

from multiagent.core import World, Action, Agent


class ScriptedAI(object):
    def __init__(self):
        pass

    def act(self, agent: Agent, world: World) -> Action:
        """
        Act based upon the world and the agent state.
        @param agent:
        @param world:
        @return:
        """
        raise NotImplementedError()


class BasicScriptedAI(ScriptedAI):

    def act(self, agent: Agent, world: World) -> Action:
        """
        Defines how a given agent should act in a given world.
        In this case the agent always targets the closest
        @param agent:
        @param world:
        @return: action
        """
        action = Action()
        action.u = np.zeros(world.dim_p + 1)
        action.u[2] = -1  # default is no target == -1

        if world.distances is not None:  # if distance matrix initialized
            masked_distances = world.distances[agent.id].copy()
            if agent.has_heal():
                mask = (world.team_mask != agent.tid) | (world.alive == 0)  # mask out all enemies and dead
            else:
                mask = (world.team_mask == agent.tid) | (world.alive == 0)  # mask out all teammates and dead

            masked_distances[mask] = np.inf  # infinite distance all non-targetable agents
            target_id = np.argmin(masked_distances)
            closest_agent = world.agents[target_id]
            distance = masked_distances[target_id]

            if distance <= agent.sight_range:  # set closest agent as target if in range
                action.u[2] = target_id  # attack >= 5 --> index 2
            else:  # move towards the closest agent if not in range
                d_pos = agent.state.pos - closest_agent.state.pos
                max_difference_dimension = np.argmax(np.abs(d_pos))
                max_diff = d_pos[max_difference_dimension]
                action.u[max_difference_dimension] = - np.sign(max_diff)  # x-axis left == 1 --> index 0
        else:  # No-Op
            action.u[0] = 0
            action.u[1] = 0

        # first two dimensions hold x and y axis movement -> multiply with movement step amount
        action.u[:2] *= world.grid_size  # (=movement step size)
        return action
