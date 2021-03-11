import math
import random


class SpawnGenerator:
    def __init__(self):
        self.generated_points = []
        self.trials = 0
        self.max_trials = 50

    def generate_team_spawns(self, cx, cy, radius):
        theta = random.uniform(0, 2 * math.pi)
        x1 = cx + radius * math.cos(theta)
        y1 = cy + radius * math.sin(theta)
        x2 = cx + radius * math.cos(-theta)
        y2 = cy + radius * math.sin(-theta)
        return [[x1, y1], [x2, y2]]

    def generate(self, cx, cy, num_points, grid_size=None, mean_radius=1.0, sigma_radius=0.1):
        points = []

        for i in range(num_points):
            point = self._generate_point(cx, cy, grid_size, mean_radius, sigma_radius)
            while point in self.generated_points: # try generating new point
                if self.trials >= self.max_trials:
                    raise Exception("Maximum trials per point reached. Try generating with more variance allowed.")
                point = self._generate_point(cx, cy, grid_size, mean_radius, sigma_radius)
                self.trials +=1
            self.generated_points.append(point)
            self.trials = 0
            points.append(point)

        return points

    def _generate_point(self, cx, cy, grid_size, mean_radius, sigma_radius):
        theta = random.uniform(0, 2 * math.pi)
        radius = random.gauss(mean_radius, sigma_radius)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        if grid_size is not None:  # Move points onto grid
            x -= x % grid_size
            y -= y % grid_size
        point = [x, y]
        return point

    def clear(self):
        self.generated_points.clear()
