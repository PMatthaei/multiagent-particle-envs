import math
import random

import numpy as np


class SpawnGenerator:
    def __init__(self, n_teams: int, world_center: np.array, grid_size=None, max_trials=50):
        """
        Generator producing random spawns. In a first step unique random spawns for each team are chosen.
        Based on these team spawn each agent receives a unique spawn.
        @param n_teams:
        @param world_bounds:
        @param grid_size:
        @param max_trials:
        """
        self.n_teams = n_teams
        self.world_center = world_center
        if self.n_teams > 2:
            raise NotImplementedError("Generating team spawns for more than two teams is not yet implemented.")
        self.grid_size = grid_size
        self.generated_points = []
        self.team_spawns = []
        self.max_trials = max_trials
        # Generate positions on grid
        self.on_grid = grid_size is not None
        if not self.on_grid:
            raise NotImplementedError("Generating spawns outside of the world grid (continuous) is not yet implemented")
        self.trials = 0

    def generate_team_spawns(self, radius):
        """
        Find team spawns on a circle with the provided radius around the map center. Teams spawn on the opposite site of
        the circle resulting in a distance of radius * 2
        @param radius:
        @return:
        """
        cx, cy = np.array(self.world_center)
        theta = random.uniform(0, 2 * math.pi)
        x1 = cx + radius * math.cos(theta)
        y1 = cy + radius * math.sin(theta)
        x2 = cx - radius * math.cos(theta)
        y2 = cy - radius * math.sin(theta)
        self.team_spawns = [np.array([x1, y1]), np.array([x2, y2])]
        norm = np.linalg.norm(self.team_spawns[0] - self.team_spawns[1])
        np.testing.assert_almost_equal(norm, radius * 2)
        return self.team_spawns

    def generate(self, cx, cy, num_points, grid_size=None, mean_radius=1.0, sigma_radius=0.1):
        """

        @param cx:
        @param cy:
        @param num_points:
        @param grid_size:
        @param mean_radius:
        @param sigma_radius:
        @return:
        """
        points = []

        for i in range(num_points):
            point = self._generate_point(cx, cy, grid_size, mean_radius, sigma_radius)
            while point in self.generated_points:  # Try generating new point if already used
                if self.trials >= self.max_trials:
                    raise Exception("Maximum trials per point reached. Try generating with more variance allowed.")
                point = self._generate_point(cx, cy, grid_size, mean_radius, sigma_radius)
                self.trials += 1
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
