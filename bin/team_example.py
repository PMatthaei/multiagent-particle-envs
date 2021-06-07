import argparse
import cProfile
import io
import pstats
import sys
from pstats import SortKey

from bin.controls.headless_controls import HeadlessControls
from bin.team_plans_example import AI_SMALL, LARGE, AI_MEDIUM, AI_LARGE, SMALL, MEDIUM, AI_VS_AI_SMALL, H2_T2_A1
from make_env import make_env
from maenv.interfaces.policy import RandomPolicy

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='teams', help='Path of the scenario Python script.')
    parser.add_argument('-p', '--profile', default=False, help='Profile the example for performance issues.')
    parser.add_argument('-bp', '--build_plan', default=H2_T2_A1, help='Build plan for the teams.')
    parser.add_argument('-stream_key', '--stream_key', default=None, help='Stream Key for Twitch.')
    parser.add_argument('-fps', '--fps', default=30, help='Locked frames per second. (Default: 30, None for unlocked.')
    args = parser.parse_args()

    env = make_env(args)

    controls = HeadlessControls(env=env)
    controls.start()

    policy_teams = env.world.policy_teams
    # create random policies for each agent in each team
    all_policies = [[RandomPolicy(env, agent) for agent in team.members] for team in policy_teams]
    # execution loop
    obs_n = env.reset()
    # render call to create viewer window (necessary only for interactive policies)
    env.render()

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
