import argparse
import logging

from bin.team_plans_example import LARGE
from multiagent.environment import MAEnv
from multiagent.interfaces.policy import RandomPolicy
from multiagent.scenarios import team

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='teams.py', help='Path of the scenario Python script.')
    parser.add_argument('-stream_key', '--stream_key', default=None, help='Stream Key for Twitch.')
    args = parser.parse_args()
    # load scenario from script
    scenario = team.load(args.scenario).TeamsScenario(LARGE)
    # create world
    world = scenario.make_teams_world(grid_size=10.0)
    # create multi-agent environment
    env = MAEnv(world=world,
                reset_callback=scenario.reset_world,
                reward_callback=scenario.reward,
                observation_callback=scenario.observation,
                info_callback=None,
                done_callback=scenario.done,
                stream_key=args.stream_key,
                headless=False,
                log_level=logging.INFO,
                log=False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create random policies for each agent in each team
    all_policies = [[RandomPolicy(env, agent) for agent in team.members] for team in world.policy_teams]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for tid, team in enumerate(world.policy_teams):
            team_policy = all_policies[tid]
            for aid, agent in enumerate(team.members):
                agent_policy = team_policy[aid]
                act_n.append(agent_policy.action())

        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        state = env.get_state()
        # render all agent views
        env.render()

        if any(done_n):
            env.reset()
