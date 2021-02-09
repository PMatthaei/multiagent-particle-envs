import random

import pygame


def hsl_to_rgb(color, alpha):
    return pygame.Color(color[0], color[1], color[2], alpha)


def generate_colors(n):
    rgb_values = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for _ in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        rgb_values.append((r, g, b))
    return rgb_values
