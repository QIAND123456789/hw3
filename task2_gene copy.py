import itertools
import random
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
plt.figure(figsize=(10, 6), dpi=80)
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

#task1结果输入
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
distances = {
    ('start', 'snacks'): 141,
    ('start', 'store'): 154,
    ('start', 'movie'): 178,
    ('start', 'food'): 218,
    
    ('snacks', 'start'): 141,
    ('snacks', 'store'): 114,
    ('snacks', 'movie'): 106,
    ('snacks', 'food'): 129,
    
    ('store', 'start'): 154,
    ('store', 'snacks'): 114,
    ('store', 'movie'): 208,
    ('store', 'food'): 110,
    
    ('movie', 'start'): 178,
    ('movie', 'snacks'): 106,
    ('movie', 'store'): 208,
    ('movie', 'food'): 111,
    
    ('food', 'start'): 218,
    ('food', 'snacks'): 129,
    ('food', 'store'): 110,
    ('food', 'movie'): 111,
    }



#task2
#初始化族群
def initialize_population(locations, population_size):
    population = []
    locs = list(locations.keys())
    locs.remove('start')  # We keep 'start' fixed for TSP
    for _ in range(population_size):
        path = locs[:]
        random.shuffle(path)
        path.insert(0, 'start')
        path.append('start')  # Start and end at 'start' location
        population.append(path)
    return population
#计算适应度
def calculate_fitness(path, distances):
    if len(path) <= 1:
        return 0  # 如果路径长度小于或等于 1，则返回 0 作为适应度
    total_distance = 0
    for i in range(len(path) - 1):
        if path[i] == path[i+1]:
            # 如果有连续重复的位置，返回一个非常小的适应度值，或者抛出一个错误
            return 0
        total_distance += distances.get((path[i], path[i+1]), None)
    if total_distance == 0:
        # 再次检查总距离是否为0，这可以帮助调试问题
        return 0
    return 1 / total_distance


#选择过程
def selection(population, fitnesses):
    # 首先，根据适应度计算选择概率
    fitness_total = sum(fitnesses)
    selection_probs = [f / fitness_total for f in fitnesses]
    # 然后，根据概率选择个体的索引
    selected_indices = np.random.choice(range(len(population)), size=len(population), p=selection_probs)
    # 最后，根据选择的索引构建新的种群
    new_population = [population[i] for i in selected_indices]
    return new_population
#交叉过程
def crossover_OX(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(1, size-1), 2))  # 避开起始和结束位置
    child = [None] * size
    # 在子代中填充来自父代1的片段
    child[start:end] = parent1[start:end]
    
    # 从父代2填充剩余位置，跳过已在子代中的元素
    position = end
    for item in parent2[end:] + parent2[1:end]:
        if item not in child:
            if position >= size:
                position = 1  # 回到起点后的第一个位置
            child[position] = item
            position += 1

    child[0] = 'start'  # 确保起点和终点是 'start'
    child[-1] = 'start'
    return child

#变异过程
def mutate(path, mutation_rate):
    path = path[1:-1]  # Remove 'start' and 'end' if they are fixed
    for i in range(len(path)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(path) - 1)
            path[i], path[j] = path[j], path[i]
    return ['start'] + path + ['start']  # Add 'start' and 'end' back if needed
#修复路径
def fix_path(path):
    # 获取路径中唯一的位置
    unique_locations = set(path[1:-1])  # 排除起始和结束位置 'start'
    # 计算缺失的位置
    missing_locations = set(locations.keys()) - unique_locations - {'start'}

    # 创建一个新的路径，避免直接修改输入的路径
    fixed_path = ['start']

    for location in path[1:-1]:  # 排除起始和结束位置 'start'
        if location in fixed_path:
            if missing_locations:
                # 如果有缺失的位置，则用一个缺失的位置替换
                fixed_path.append(missing_locations.pop())
            else:
                # 如果没有缺失的位置可用（理论上不应该发生），打印错误或采取其他措施
                print("Error: No missing location to replace duplicate.")
        else:
            fixed_path.append(location)

    fixed_path.append('start')  # 重新添加结束位置 'start'
    return fixed_path



# 遗传算法的主函数
# 遗传算法的主函数
def genetic_algorithm(locations, distances, population_size, generations, mutation_rate):
    population = initialize_population(locations, population_size)
    for _ in range(generations):
        fitnesses = [calculate_fitness(p, distances) for p in population]

        # 选择
        new_population = selection(population, fitnesses)

        # 交叉和变异
        children = []
        for i in range(0, len(new_population), 2):
            parent1, parent2 = new_population[i], new_population[i + 1]
            
            # 应用交叉
            child1 = crossover_OX(parent1, parent2)
            child2 = crossover_OX(parent2, parent1)
            
            # 应用变异
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            # 修复路径 - 确保路径有效
            child1 = fix_path(child1)
            child2 = fix_path(child2)
            
            children.extend([child1, child2])
        
        population = children

    # 找到并返回最佳路径
    best_fitness = max([calculate_fitness(p, distances) for p in population])
    best_path = fitnesses.index(best_fitness)
    return population[best_path]
# Run the genetic algorithm to get the best path
best_path = genetic_algorithm(locations, distances, population_size=100, generations=1000, mutation_rate=0.01)
# Now let's run the GA
#best_path = genetic_algorithm(locations, distances, population_size=100, generations=1000, mutation_rate=0.01)

# Visualization or other use of best_path
def visualize_path(best_path, locations, grid_map):
    plt.figure(figsize=(10, 6))
    plt.imshow(grid_map_img, cmap='gray')
    plot_locations(locations, color='cyan')
    
    # Iterate through the best path and use A* to find the actual path
    for i in range(len(best_path) - 1):
        start = best_path[i]
        end = best_path[i + 1]
        start_coords = locations[start]
        end_coords = locations[end]
        
        # Convert the location coordinates to grid indices
        start_grid = (start_coords[0], start_coords[1])  
        end_grid = (end_coords[0], end_coords[1])        
        # Get the actual path from the A* algorithm
        actual_path = a_star(start_grid, end_grid, grid_map)
        
        
        # Plot the actual path
        for j in range(len(actual_path) - 1):
            plt.plot(
                [actual_path[j][0], actual_path[j+1][0]],  
                [ actual_path[j][1], actual_path[j+1][1]],  'ro-'  )
    
    plt.show()




# Visualize the path using the A* algorithm for pathfinding
visualize_path(best_path, locations, grid_map)
print("Best Path:", best_path)