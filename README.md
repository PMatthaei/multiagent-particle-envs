[![Python package](https://github.com/PMatthaei/ma-env/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/PMatthaei/ma-env/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/PMatthaei/ma-env/branch/master/graph/badge.svg?token=YMA67UWT20)](https://codecov.io/gh/PMatthaei/ma-env)
```
This is a fork of: https://github.com/openai/multiagent-particle-envs
```

# Teamfight(RTS/MMO) Micro-Management Multi-Agent Environment

A simple multi-agent environment, allowing for team-based micro-management tasks. Most prominent features:
- Role-based agents (f.e. healer, tank, etc)
- Heterogeneous/homogeneous and symmetric/asymmetric team compositions
- Rendering via PyGame (optional headless)
- Recording via ffmpeg (optional headless)
- Streaming via twitch

## Example:

Look at the [example](https://github.com/PMatthaei/ma-env/blob/master/bin/team_example.py) to learn basic usage of the environment.

## Getting started:

- To install, `cd` into the root directory and type `pip install -r requirements.txt`

- To start a simple example run: `bin/team_example.py`

- To use recording _(default=true)_, install ffmpeg: 
  - Linux: `sudo apt-get install ffmpeg`
  - Windows: [Download Binaries](https://github.com/BtbN/FFmpeg-Builds/releases) and set in `PATH`.

The following ffmpeg version was used:
```
ffmpeg version 4.2.4-1ubuntu0.1 Copyright (c) 2000-2020 the FFmpeg developers
  built with gcc 9 (Ubuntu 9.3.0-10ubuntu2)
  configuration: --prefix=/usr --extra-version=1ubuntu0.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-avresample --disable-filter=resample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-nvenc --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 31.100 / 56. 31.100
  libavcodec     58. 54.100 / 58. 54.100
  libavformat    58. 29.100 / 58. 29.100
  libavdevice    58.  8.100 / 58.  8.100
  libavfilter     7. 57.100 /  7. 57.100
  libavresample   4.  0.  0 /  4.  0.  0
  libswscale      5.  5.100 /  5.  5.100
  libswresample   3.  5.100 /  3.  5.100
  libpostproc    55.  5.100 / 55.  5.100
```
## Code structure

- `make_env.py`: contains code for importing a multiagent environment as an OpenAI Gym-like object.

- `./maenv/environment.py`: contains code for environment simulation (interaction physics, `_step()` function, etc.)

- `./maenv/core.py`: contains classes for various objects (Entities, Game Objects, Agents, etc.) that are used throughout the code.

- `./maenv/pygame_rendering.py`: used for displaying agent behaviors on the screen.

- `./maenv/policy.py`: contains code for interactive policy based on keyboard input.

- `./maenv/scenario.py`: contains base scenario object that is extended for all scenarios.

- `./maenv/interfaces/scenario`: folder where various scenarios/ environments are stored. scenario code consists of several functions:
    1) `make_teams_world()`: creates all of the entities that inhabit the world, assigns their capabilities and team affiliation. called once at the beginning of each training session
    
    `_build_teams`: called internally via `make_teams_world`. defines team affiliations of entities.
    
    `_make_world`: called internally via `make_teams_world`. creates world and entity capabilities.
    
    2) `reset_world()`: resets the world by assigning properties (position, color, etc.) to all entities in the world
    called before every episode (including after make_world() before the first episode)
    3) `reward()`: defines the reward function for a given agent
    4) `observation()`: defines the observation space of a given agent
    5) (optional) `benchmark_data()`: provides diagnostic data for policies trained on the environment (e.g. evaluation metrics)

### Creating new environments

You can create new scenarios by implementing the first 5 functions above (`make_world()`, `build_teams`, `reset_world()`, `reward()`, and `observation()`).

## Internals

### Why numpy 1.16?

See performance problems with [`numpy.core._multiarray_umath.implement_array_function`](https://stackoverflow.com/questions/58909525/what-is-numpy-core-multiarray-umath-implement-array-function-and-why-it-costs-l)
