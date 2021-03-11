import math
import random


def generate_spawns(cx, cy, num_points, grid_size=None, mean_radius=1.0, sigma_radius=0.1):
    points = []
    for i in range(num_points):
        theta = random.uniform(0, 2 * math.pi)
        radius = random.gauss(mean_radius, sigma_radius)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        if grid_size is not None:  # Move points onto grid
            x -= x % grid_size
            y -= y % grid_size
        points.append([x, y])
    return points
