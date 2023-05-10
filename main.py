from PIL import Image
import numpy as np
import itertools
import imageio
from heapq import heappop, heappush
import random


def create_gif(mask_path, real_path, max_combinations=0, return_to_origin=False, dot_size=0.01, dot_speed=1,
               wait_time=(5, 10), dot_color=(255, 255, 0)):
    mask = Image.open(mask_path).convert('RGB')
    mask_data = np.array(mask)
    walkable = np.logical_or(np.all(mask_data == [255, 0, 0], axis=2), np.all(mask_data == [0, 0, 255], axis=2))
    origins = np.all(mask_data == [0, 0, 255], axis=2)

    real = Image.open(real_path).convert('RGB')
    real_data = np.array(real)

    origin_coords = np.argwhere(origins)
    combinations = list(itertools.combinations(origin_coords, 2))
    if max_combinations > 0:
        max_combinations = min(max_combinations, len(combinations))
        combinations = random.sample(combinations, max_combinations)

    def a_star(start, goal):
        # Heuristic function
        def h(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Initialize open and closed sets
        open_set = [(h(start), start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            current = heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in [(current[0] + d[0], current[1] + d[1]) for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]]:
                if not (0 <= neighbor[0] < walkable.shape[0] and 0 <= neighbor[1] < walkable.shape[1]):
                    continue
                if not walkable[neighbor]:
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + h(neighbor)
                    heappush(open_set, (f_score, neighbor))

        return []

    # Find paths for all combinations
    paths = []
    max_path_length = 0
    for origin_index in range(len(origin_coords)):
        origin = origin_coords[origin_index]
        dest_index = (origin_index + random.randint(1, len(origin_coords) - 1)) % len(origin_coords)
        dest = origin_coords[dest_index]
        path = a_star(tuple(origin), tuple(dest))
        if return_to_origin:
            # Add delay before returning to origin
            delay = random.randint(*wait_time)
            path += [dest] * delay
            path += a_star(tuple(dest), tuple(origin))[1:]
            # Add delay before starting again
            delay = random.randint(*wait_time)
            path += [origin] * delay
        paths.append(path)
        max_path_length = max(max_path_length, len(path))

    # Compute dot size
    dot_radius = int(max(mask_data.shape) * dot_size)
    dot_diameter = dot_radius * 2 + 1

    # Create gif frames
    frames = []
    for i in range(0, max_path_length):
        frame = real_data.copy()
        for path_index in range(len(paths)):
            path = paths[path_index]
            point_index = min(i // dot_speed, len(path) - 1)
            point = path[point_index]
            frame[max(0, point[0] - dot_radius):min(frame.shape[0], point[0] + dot_radius + 1),
            max(0, point[1] - dot_radius):min(frame.shape[1], point[1] + dot_radius + 1)] = dot_color
        frames.append(frame)

    # Save gif
    imageio.mimsave('output.gif', frames)


create_gif('mask.png', 'real.png', max_combinations=0,
           return_to_origin=True,
           dot_speed=1,
           dot_size=0.0025,
           wait_time=(25, 50),
           dot_color=(0, 255, 0))
