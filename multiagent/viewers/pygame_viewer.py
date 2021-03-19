from __future__ import annotations

import math
import os

import subprocess as sp

import numpy as np
import pygame
from pygame.rect import Rect

from multiagent.core import RoleTypes, Agent
from multiagent.utils.colors import tuple_to_color, colour_to_color
from multiagent.viewers.twitch_viewer import TwitchViewer
from colour import Color

HEALTH_BAR_HEIGHT = 2
HEALTH_BAR_WIDTH = 15
HEALTH_BAR_COLOR = tuple(c / 255.0 for c in [102, 171, 79])
MISSING_HEALTH_BAR_COLOR = (61, 61, 61)
HEALTH_BAR_COLOR_RANGE = list(Color(rgb=HEALTH_BAR_COLOR).range_to(Color("red"), 3))
HEALTH_BAR_COLOR_RANGE_N = len(HEALTH_BAR_COLOR_RANGE)

def check_ffmpeg():
    ffmpeg_available = True
    print('Check if ffmpeg is installed...')
    try:
        print(sp.check_output(['which', 'ffmpeg']))
    except Exception as e:
        print(e)
        ffmpeg_available = False
    if not ffmpeg_available:
        print("Could not find ffmpeg. Please run 'sudo apt-get install ffmpeg'.")
    else:
        print('ffmpeg is installed.')

    return ffmpeg_available


class _Grid:
    def __init__(self, screen, cell_size):
        self.screen = screen
        self.surface = screen.convert_alpha()
        self.surface.fill([0, 0, 0, 0])
        self.col_n = math.ceil(screen.get_width() / cell_size)
        self.line_n = math.ceil(screen.get_height() / cell_size)
        self.cell_size = cell_size
        self.grid = [[0 for i in range(self.col_n)] for j in range(self.line_n)]

    def draw_use_line(self):
        for li in range(self.line_n):
            li_coord = li * self.cell_size
            pygame.draw.line(self.surface, (0, 0, 0, 50), (0, li_coord), (self.surface.get_width(), li_coord))
        for co in range(self.col_n):
            colCoord = co * self.cell_size
            pygame.draw.line(self.surface, (0, 0, 0, 50), (colCoord, 0), (colCoord, self.surface.get_height()))

        self.screen.blit(self.surface, (0, 0))


class _SpriteFactory:
    @staticmethod
    def build(agent: Agent):
        if RoleTypes.TANK in agent.unit_id:
            return _Tank(agent)
        elif RoleTypes.ADC in agent.unit_id:
            return _ADC(agent)
        elif RoleTypes.HEALER in agent.unit_id:
            return _Healer(agent)
        else:
            raise Exception()


class PyGameViewer(object):
    def __init__(self,
                 env,
                 caption="Multi-Agent Environment",
                 fps=30,
                 infos=True,
                 draw_grid=True,
                 record=False,
                 stream_key=None,
                 headless=True):
        """
        Create new PyGameViewer for the environment
        :param env: MultiAgentEnv to render
        :param caption: Caption of the window
        :param fps: Frames per second
        :param infos: Show additional information about performance and current game state.
        :param draw_grid: Draw underlying movement grid induced by step size
        :param record: Activate recording
        :param headless: Make rendering headless (no window)
        """
        self.env = env
        self.entities = None
        self.draw_grid = draw_grid
        self.record = record
        self.stream = stream_key is not None
        self.proc = None
        self.headless = headless

        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Skip audio module since it is not used and produced errors with ALSA lib on ubuntu
        pygame.display.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('', 25)
        flags = pygame.DOUBLEBUF

        self.screen = pygame.display.set_mode(self.env.world.bounds, flags=flags)

        # Improve event queue with restricted events
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.K_ESCAPE])

        if self.draw_grid:
            self.grid = _Grid(screen=self.screen, cell_size=int(self.env.world.grid_size))

        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.dt = 0
        self.fps = fps
        self.infos = infos
        self.clear()

        width, height = self.env.world.bounds

        if self.record and check_ffmpeg():
            self.proc = sp.Popen(['ffmpeg',
                                  '-y',
                                  '-f', 'rawvideo',
                                  '-vcodec', 'rawvideo',
                                  '-s', str(width) + 'x' + str(height),
                                  '-pix_fmt', 'rgba',
                                  '-r', str(self.fps),
                                  '-i', '-',
                                  '-an',
                                  'env-recording.mov'], stdin=sp.PIPE)
        elif self.stream and check_ffmpeg():
            self.twitch = TwitchViewer(stream_key=stream_key,
                                       width=width, height=height)
        pass

    def update(self):
        """
        Update data. This does not update visuals
        :param t:
        :param episode:
        :return:
        """

        if self.infos:
            dt = self.font.render("FPS: " + str(self.fps), False, (0, 0, 0))
            t = self.font.render("Time step: " + str(self.env.t), False, (0, 0, 0))
            episode = self.font.render("Episode: " + str(self.env.episode), False, (0, 0, 0))
            max_step = self.font.render("Max. Step: " + str(self.env.episode_limit), False, (0, 0, 0))
            self.screen.blit(dt, (0, 0))
            self.screen.blit(t, (0, 20))
            self.screen.blit(episode, (0, 40))
            self.screen.blit(max_step, (0, 60))

        if self.draw_grid:
            self.grid.draw_use_line()

        # update entity positions visually
        for entity in self.entities:
            entity.update()
            if entity.is_dead():
                self.entities.remove(entity)
            else:
                self.screen.blit(entity.surf, entity.rect)

    def init(self, world_entities):
        """
        Initialize viewer with its rendered entities, created from their counterparts in the environment data.
        :param world_entities:
        :return:
        """
        self.entities = pygame.sprite.Group()
        self.entities.add(*[_SpriteFactory.build(entity) for entity in world_entities])

    def render(self):
        if self.headless:
            return
        """
        Render current data and handle events
        :return:
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.display.quit()
                    pygame.quit()
                    exit()

        pygame.display.flip()

        if self.record:
            self.proc.stdin.write(self.screen.get_buffer())
        elif self.stream:
            frame = pygame.surfarray.array3d(self.screen)
            self.twitch.send_frame(np.true_divide(frame, 255))

        self.dt = self.clock.tick(self.fps)

    def reset(self):
        """
        Reset the visuals to default
        :return:
        """
        self.entities = None
        self.clear()

    def clear(self):
        """
        Clear screen. Usually called to clear screen for next frame.
        :return:
        """
        self.screen.fill((255, 255, 255))

    def close(self):
        self.entities = None
        pygame.quit()
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc.terminate()


class _PyGameEntity(pygame.sprite.Sprite):
    def __init__(self, agent: Agent, debug_range=False, debug_health=True):
        super(_PyGameEntity, self).__init__()
        self.agent = agent  # This reference is updated in world step
        self.debug_range = debug_range
        self.debug_health = debug_health
        self.color = self.agent.color
        self.sight_range = self.agent.sight_range
        self.attack_range = self.agent.attack_range
        self.body_radius = self.agent.bounding_circle_radius
        self.alpha = 255
        self.surf = pygame.Surface((self.sight_range * 2, self.sight_range * 2), pygame.SRCALPHA, 32).convert_alpha()
        self.rect: Rect = self.surf.get_rect()
        self.update()

    def is_dead(self):
        return self.agent.is_dead()

    def update(self):
        # Only redraw if visible state changed
        self._draw()
        # Important: The simulation is updating with a move_by update while here we set the resulted new pos
        # TODO: if move_by needed the delta update needs to be saved in entity so we can recreate it here visually
        self.rect.centerx = self.agent.state.pos[0]
        self.rect.centery = self.agent.state.pos[1]

    def _draw(self):
        self.alpha = 80 if self.agent.is_dead() else 255
        if self.debug_health:
            self._draw_health_bar()
        if self.debug_range:
            self._draw_ranges()

    def _draw_health_bar(self):
        bar_width = self.body_radius * 2 + 2
        rel_health = self.agent.state.health / self.agent.state.max_health
        health_bar = bar_width * rel_health
        missing_rel_health = (1.0 - rel_health)
        missing_health_bar = bar_width * missing_rel_health

        center_x = self.sight_range - bar_width / 2.0
        center_y = self.sight_range - HEALTH_BAR_HEIGHT / 2.0

        health_bar_color = self._get_health_color(missing_rel_health)
        color = colour_to_color(health_bar_color, self.alpha)
        health_bar_rect = Rect(center_x, center_y - self.body_radius - HEALTH_BAR_HEIGHT, health_bar, HEALTH_BAR_HEIGHT)

        pygame.draw.rect(self.surf, color=color, rect=health_bar_rect)

        missing_health_color = tuple_to_color(MISSING_HEALTH_BAR_COLOR, self.alpha)
        missing_health_bar = Rect(center_x + health_bar, center_y - self.body_radius - HEALTH_BAR_HEIGHT,
                                  missing_health_bar,
                                  HEALTH_BAR_HEIGHT)
        pygame.draw.rect(self.surf, color=missing_health_color, rect=missing_health_bar)

    def _get_health_color(self, missing_rel_health):
        health_category = math.ceil(missing_rel_health / (1 / 3))
        color_index = np.clip(health_category, 0, HEALTH_BAR_COLOR_RANGE_N - 1)
        health_bar_color = HEALTH_BAR_COLOR_RANGE[color_index]
        return health_bar_color

    def _draw_ranges(self):
        center = (self.sight_range, self.sight_range)
        color = tuple_to_color(self.color, self.alpha)
        pygame.draw.circle(self.surf, color=color, center=center, radius=self.sight_range, width=1)
        pygame.draw.circle(self.surf, color=color, center=center, radius=self.attack_range, width=1)


class _ADC(_PyGameEntity):

    def _draw(self):
        super(_ADC, self)._draw()
        center = (self.sight_range, self.sight_range)
        color = tuple_to_color(self.color, self.alpha)
        pygame.draw.circle(self.surf, color=color, center=center, radius=self.body_radius)


class _Healer(_PyGameEntity):

    def _draw(self):
        super(_Healer, self)._draw()
        c = self.sight_range
        w = (self.body_radius * 2) / 3.0
        h = w * 3
        color = tuple_to_color(self.color, self.alpha)
        pygame.draw.rect(self.surf, color=color, rect=Rect(c - w / 2, c - h / 2, w, h))
        pygame.draw.rect(self.surf, color=color, rect=Rect(c - h / 2, c - w / 2, h, w))


class _Tank(_PyGameEntity):

    def _draw(self):
        super(_Tank, self)._draw()
        left, top = (self.sight_range - self.body_radius,) * 2
        width, height = (self.body_radius * 2,) * 2
        rect = Rect(left, top, width, height)
        color = tuple_to_color(self.color, self.alpha)
        pygame.draw.rect(self.surf, color=color, rect=rect)
