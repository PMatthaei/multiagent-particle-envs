import argparse
import cProfile
import io
import logging
import pstats
import sys
import threading
import time
from pstats import SortKey

from bin.interactive import EnvControls
from bin.team_plans_example import AI_SMALL
from make_env import make_env
from multiagent.interfaces.policy import RandomPolicy

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='teams', help='Path of the scenario Python script.')
    parser.add_argument('-p', '--profile', default=False, help='Profile the example for performance issues.')
    parser.add_argument('-bp', '--build_plan', default=AI_SMALL, help='Build plan for the teams.')
    parser.add_argument('-stream_key', '--stream_key', default=None, help='Stream Key for Twitch.')
    args = parser.parse_args()

    env = make_env(args)

    controls = EnvControls(env=env)
    controls.start()

    # render call to create viewer window (necessary only for interactive policies)
    env.render()

    policy_teams = env.world.policy_teams
    # create random policies for each agent in each team
    all_policies = [[RandomPolicy(env, agent) for agent in team.members] for team in policy_teams]
    # execution loop
    obs_n = env.reset()

    profiler = None
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        while True:
            # query for action from each agent's policy
            act_n = []
            for tid, team in enumerate(policy_teams):
                team_policy = all_policies[tid]
                for aid, agent in enumerate(team.members):
                    agent_policy = team_policy[aid]
                    act_n.append(agent_policy.action())

            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            state = env.get_state()
            # render all agent views
            env.render()

            if args.profile and profiler:
                s = io.StringIO()
                sortby = SortKey.TIME
                ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
                ps.print_stats()
                print(s.getvalue())

            if any(done_n):
                env.reset()
    except KeyboardInterrupt:
        controls.stop()
        controls.join()
        sys.exit()
