import itertools
import numpy as np
import imageio
import matplotlib.pyplot as plt
from queue import PriorityQueue
# matplotlib inline

# Load the map
floor_plan = imageio.imread('./map/vivocity.png')
grid_map_img = imageio.imread('./map/vivocity_freespace.png')
grid_map = grid_map_img.transpose()
print("Size of the map:", grid_map.shape)
print("Occupied Cells:", np.count_nonzero(grid_map == 0))
print("Free Cells:", np.count_nonzero(grid_map == 255))

# Map resolution (Constant)
MAP_RES = 0.2 # each cell represents a 0.2m x 0.2m square in reality

# Locations on the map
locations = {'start':  [345, 95],    # Start from the level 2 Escalator
             'snacks': [470, 475],   # Garrett Popcorn
             'store':  [20, 705],    # DJI Store
             'movie':  [940, 545],   # Golden Village
             'food':   [535, 800],   # PUTIEN
            }

# A helper function to mark the locations on the map
def plot_locations(locations: dict, color: 'str'='black'):
    for key, value in locations.items():
        plt.plot(locations[key][0], locations[key][1], marker="o", markersize=10, markeredgecolor="red")
        plt.text(locations[key][0], locations[key][1]-15, s=key, fontsize='x-large', fontweight='bold', c=color, ha='center')
    return

# Plot the locaitons on the map 
plt.figure(figsize=(20, 10), dpi=80)
plt.subplot(1,2,1)
plt.imshow(floor_plan)
plot_locations(locations)
plt.subplot(1,2,2)
plt.imshow(grid_map_img, cmap='gray')
plot_locations(locations, color='cyan')
plt.show()

# Check if the designated locations are free
for key, value in locations.items():
    print(f"Cell {key} is free: {grid_map[value[0], value[1]] == 255}")

#task1，A*
def heuristic(a, b):
    # 使用曼哈顿距离作为启发式函数
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def get_neighbors(node, grid_map):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # 8方向
        x, y = node[0] + dx, node[1] + dy
        if 0 <= x < grid_map.shape[0] and 0 <= y < grid_map.shape[1] and grid_map[x, y] == 255: # 检查是否在边界内且为自由格
            neighbors.append((x, y))
    return neighbors
def a_star(start, goal, grid_map):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start) # optional
            path.reverse() # optional
            return path

        for neighbor in get_neighbors(current, grid_map):
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                open_set.put((f_score[neighbor], neighbor))

    return []


#task2
locations_pairs = list(itertools.permutations(locations.keys(), 2))
distances = {}

for pair in locations_pairs:
    start, goal = pair
    start_coords = tuple(locations[start])
    goal_coords = tuple(locations[goal])
    path = a_star(start_coords, goal_coords, grid_map)
    distances[pair] = len(path) * MAP_RES  # 假设每步代表0.2m

def nearest_neighbor(start, distances):
    unvisited = set(locations.keys())
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        next_destination = min(unvisited, key=lambda x: distances[(current, x)])
        unvisited.remove(next_destination)
        tour.append(next_destination)
        current = next_destination
    tour.append(start) # 回到起点
    return tour

tour = nearest_neighbor('start', distances)

# 可视化
plt.imshow(grid_map_img, cmap='gray')
plot_locations(locations, color='cyan')

# 绘制路径
# 在可视化之前将路径坐标转置
path_transposed = [(y, x) for x, y in path]
# 绘制路径
for i in range(len(tour) - 1):
    start, end = tour[i], tour[i+1]
    path = a_star(tuple(locations[start]), tuple(locations[end]), grid_map)
    path_transposed = [(y, x) for x, y in path]  # 转置路径坐标
    for point in path_transposed:
        plt.plot(point[1], point[0], 'ro', markersize=2)
plt.show()
