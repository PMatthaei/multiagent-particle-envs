import random

import colour
import pygame


def colour_to_color(color: colour.Color, alpha):
    tupled_color = tuple([int(255 * c) for c in color.rgb])
    return tuple_to_color(tupled_color, alpha)


def tuple_to_color(color: tuple, alpha):
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


def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c


def complement(r, g, b):
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))
