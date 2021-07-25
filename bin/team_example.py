import cProfile
import io
import pstats
import sys
from os.path import abspath, dirname
from pstats import SortKey

from bin.controls.headless_controls import HeadlessControls
from bin.team_plans_example import *
from core import RoleTypes
from make_env import make_env
from maenv.interfaces.policy import RandomPolicy
import nestargs

if __name__ == '__main__':

    # parse arguments
    parser = nestargs.NestedArgumentParser()

    parser.add_argument('--scenario', default='teams', help='Path of the scenario Python script.')
    parser.add_argument('--profile', default=False, help='Profile the example for performance issues.')
    parser.add_argument('--stream_key', default=None, help='Stream Key for Twitch.')

    parser.add_argument('--scenario_args.match_build_plan', default=ALL, help='Build plan for the teams.')
    parser.add_argument('--scenario_args.grid_size', default=20, help='Edge length of a grid cell. Step size of a unit.')
    parser.add_argument('--scenario_args.random_spawns', default=False, help='')
    parser.add_argument('--scenario_args.stochastic_spawns', default=True, help='')
    parser.add_argument('--scenario_args.ai', default='basic', help='')
    parser.add_argument('--scenario_args.ai_config', default={}, help='')
    parser.add_argument('--scenario_args.attack_range_only', default=False, help='')

    parser.add_argument('--viewer_args.fps', default=60, help='')
    parser.add_argument('--viewer_args.headless', default=True, help='')
    parser.add_argument('--viewer_args.record', default=dirname(abspath(__file__)), help='')
    parser.add_argument('--viewer_args.debug_health', default=True, help='')
    parser.add_argument('--viewer_args.debug_range', default=True, help='')

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
            mask = env.get_mask()
            entities = env.get_entities()
            print(reward_n)
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
