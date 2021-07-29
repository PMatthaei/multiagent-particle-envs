from gym.envs.registration import register
from .make_env import make_env
# Multiagent envs
# ----------------------------------------

register(
    id='MultiagentSimple-v0',
    entry_point='maenv.envs:SimpleEnv',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)

register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='maenv.envs:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)
