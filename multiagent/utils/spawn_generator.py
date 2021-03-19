import math
import random

import numpy as np

from multiagent.core import World


class SpawnGenerator:
    def __init__(self, world: World, num_pos: int, max_trials=50):
        """
        Generator producing random spawns. In a first step unique random spawns for each team are chosen.
        Based on these team spawn each agent receives a unique spawn.
        @param world_bounds:
        @param grid_size:
        @param max_trials:
        """
        self.n_teams = len(world.teams)
        self.world_center = world.grid_center
        if self.n_teams > 2:
            raise NotImplementedError("Generating team spawns for more than two teams is not yet implemented.")
        self.grid_size = world.grid_size
        self.used_points = np.full((num_pos, world.dim_p),fill_value=np.inf)
        self.team_spawns = []
        self.max_trials = max_trials
        # Generate positions on grid
        self.on_grid = self.grid_size is not None
        if not self.on_grid:
            raise NotImplementedError("Generating spawns outside of the world grid (continuous) is not yet implemented")
        self.trials = 0

    def generate_team_spawns(self, radius):
        """
        Generate team spawns on a circle with the provided radius around the world center.
        Teams spawn on the opposite site of the circle resulting in a distance of radius * 2.
        Team spawns are always real numbered.
        @param radius: defined circle on which team spawns are generated
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
        # Ensure integrity of function in case of changes
        np.testing.assert_almost_equal(norm, radius * 2)
        np.testing.assert_(len(self.team_spawns) == 2)
        return self.team_spawns

    def generate(self, center: np.array, num_points: int, grid_size=None, mean_radius=1.0, sigma_radius=0.1):
        points = []

        for i in range(num_points):
            point = self._generate_point(center, mean_radius, sigma_radius, grid_size=grid_size)
            used = np.all(self.used_points[:, [0, 1]] == point, axis=1)
            while np.any(used):  # Try generating new point if already used
                if self.trials >= self.max_trials:
                    raise Exception("Maximum trials per point reached. Try generating with more variance allowed.")
                point = self._generate_point(center, mean_radius, sigma_radius, grid_size=grid_size)
                used = np.all(self.used_points[:, [0, 1]] == point, axis=1)
                self.trials += 1
            self.used_points[i] = point
            self.trials = 0
            points.append(point)

        return points

    def _generate_point(self, center: np.array, mean_radius: float, sigma_radius: float, grid_size: int = None):
        """
        Generates a point around the center with a mean radius.
        @param center:
        @param grid_size:
        @param mean_radius:
        @param sigma_radius:
        @return:
        """
        theta = random.uniform(0, 2 * math.pi)
        radius = random.gauss(mean_radius, sigma_radius)  # noise
        point = center + radius * np.array([math.cos(theta), math.sin(theta)])
        if grid_size is not None:  # Move points onto grid
            point -= (point % grid_size)
        return point

    def clear(self):
        self.used_points = None
