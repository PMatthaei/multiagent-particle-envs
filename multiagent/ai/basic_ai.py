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
    """
                agent.action.u[0] = -1.0  # x-axis left == 1 --> index 0
        elif act_ind == 2:
            agent.action.u[0] = +1.0  # x-axis right == 2 --> index 0
        elif act_ind == 3:
            agent.action.u[1] = -1.0  # y-axis up == 3 --> index 1
        elif act_ind == 4:
            agent.action.u[1] = +1.0  # y-axis down == 4 --> index 1
    """

    def act(self, agent: Agent, world: World) -> Action:
        action = Action()
        action.u = np.zeros(world.dim_p + 1)
        action.u[2] = -1  # default is no target == -1

        if len(world.distance_matrix) > 0:  # if distance matrix initialized
            masked_distances = world.distance_matrix[agent.id].copy()
            if agent.has_heal():
                mask = [a.tid != agent.tid for a in world.agents]  # mask out all enemies
            else:
                mask = [a.tid == agent.tid for a in world.agents]  # mask out all teammates

            masked_distances[mask] = np.inf
            closest_id = np.argmin(masked_distances)
            closest_agent = world.agents[closest_id]
            distance = masked_distances[closest_id]

            if distance <= agent.sight_range:  # set enemy as target if in range
                action.u[2] = closest_id  # attack >= 5 --> index 2
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
