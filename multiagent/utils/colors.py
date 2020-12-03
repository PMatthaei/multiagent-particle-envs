import pygame


def hsl_to_rgb(color, alpha):
    color = tuple(int(c * 255) for c in color)
    return pygame.Color(color[0], color[1], color[2], alpha)
