import math
import os

import subprocess as sp
import pygame
from pygame.rect import Rect

from multiagent.core import Entity, RoleTypes, Agent
from multiagent.utils.colors import hsl_to_rgb

HEALTH_BAR_HEIGHT = 4

HEALTH_BAR_WIDTH = 25


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


class Grid:
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


class PyGameViewer(object):
    def __init__(self, env, caption="Multi-Agent Environment", fps=30, infos=True, draw_grid=True, record=True,
                 headless=False):
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
        self.proc = None
        self.fps = 30
        self.headless = headless

        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Skip audio module since it is not used and produced errors with ALSA lib on ubuntu
        pygame.display.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('', 25)

        self.screen = pygame.display.set_mode(self.env.world.bounds)
        if self.draw_grid:
            self.grid = Grid(screen=self.screen, cell_size=int(self.env.world.grid_size))

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
            max_step = self.font.render("Max. Step: " + str(self.env.max_steps), False, (0, 0, 0))
            self.screen.blit(dt, (0, 0))
            self.screen.blit(t, (0, 20))
            self.screen.blit(episode, (0, 40))
            self.screen.blit(max_step, (0, 60))

        if self.draw_grid:
            self.grid.draw_use_line()

        # update entity positions visually
        for entity in self.entities:
            entity.update()
            self.screen.blit(entity.surf, entity.rect)

    def init(self, world_entities):
        """
        Initialize viewer with its rendered entities, created from their counterparts in the environment data.
        :param world_entities:
        :return:
        """
        self.entities = pygame.sprite.Group()
        self.entities.add(*[PyGameEntity(entity) for entity in world_entities])

    def render(self):
        """
        Render current data and handle events
        :return:
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pygame.display.flip()

        if self.record:
            self.proc.stdin.write(self.screen.get_buffer())

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
        pygame.quit()
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()


class PyGameEntity(pygame.sprite.Sprite):
    def __init__(self, agent: Agent):
        super(PyGameEntity, self).__init__()
        # This reference is updated in world step
        self.agent = agent
        radius = agent.sight_range
        # This is its visual representation
        self.surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA, 32).convert_alpha()
        self.rect: Rect = self.surf.get_rect()
        # Move to initial position
        self.draw()
        self.update()

    def update(self):
        # Only redraw if visible state changed
        if self.agent.is_dead():
            self.draw()
        # Important: The simulation is updating with a move_by update while here we set the resulted new pos
        # TODO: if move_by needed the update needs to be saved in entity so we can recreate it here visually
        self.rect.centerx = self.agent.state.pos[0]
        self.rect.centery = self.agent.state.pos[1]

    def draw(self):
        alpha = 80 if self.agent.is_dead() else 255
        color = self.agent.color
        sight_range = self.agent.sight_range
        attack_range = self.agent.attack_range
        body_radius = self.agent.bounding_circle_radius
        if RoleTypes.TANK in self.agent.role:
            self.draw_tank(alpha, body_radius, color, sight_range)
        elif RoleTypes.HEALER in self.agent.role:
            self.draw_healer(alpha, body_radius, color, sight_range)
        else:
            self.draw_adc(alpha, body_radius, color, sight_range)
        self.draw_health_bar(alpha, body_radius, color, sight_range)
        self.draw_ranges(alpha, attack_range, color, sight_range)

    def draw_health_bar(self, alpha, body_radius, color, sight_range):
        rel_health = self.agent.state.health / self.agent.state.max_health
        health_bar = HEALTH_BAR_WIDTH * rel_health
        missing_health = HEALTH_BAR_WIDTH * (1 - rel_health)
        center_x = sight_range - HEALTH_BAR_WIDTH / 2.0
        center_y = sight_range - HEALTH_BAR_HEIGHT / 2.0
        pygame.draw.rect(self.surf, color=hsl_to_rgb(color, alpha),
                         rect=Rect(center_x, center_y - body_radius - HEALTH_BAR_HEIGHT, health_bar,
                                   HEALTH_BAR_HEIGHT))
        pygame.draw.rect(self.surf, color=hsl_to_rgb((61, 61, 61), alpha),
                         rect=Rect(center_x + health_bar, center_y - body_radius - HEALTH_BAR_HEIGHT, missing_health,
                                   HEALTH_BAR_HEIGHT))

    def draw_ranges(self, alpha, attack_range, color, sight_range):
        pygame.draw.circle(self.surf, color=hsl_to_rgb(color, alpha), center=(sight_range, sight_range),
                           radius=sight_range, width=1)
        pygame.draw.circle(self.surf, color=hsl_to_rgb(color, alpha), center=(sight_range, sight_range),
                           radius=attack_range, width=1)

    def draw_adc(self, alpha, body_radius, color, sight_range):
        pygame.draw.circle(self.surf, color=hsl_to_rgb(color, alpha), center=(sight_range, sight_range),
                           radius=body_radius)

    def draw_tank(self, alpha, body_radius, color, sight_range):
        rect = Rect(sight_range - body_radius, sight_range - body_radius, body_radius * 2, body_radius * 2)
        pygame.draw.rect(self.surf, color=hsl_to_rgb(color, alpha), rect=rect)

    def draw_healer(self, alpha, body_radius, color, sight_range):
        c = sight_range
        w = (body_radius * 2) * 1.0 / 3.0
        h = w * 3
        pygame.draw.rect(self.surf, color=hsl_to_rgb(color, alpha), rect=Rect(c - w / 2, c - h / 2, w, h))
        pygame.draw.rect(self.surf, color=hsl_to_rgb(color, alpha), rect=Rect(c - h / 2, c - w / 2, h, w))
