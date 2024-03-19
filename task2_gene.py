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
#task1结果输入
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


# Now let's run the GA
best_path = genetic_algorithm(locations, distances, population_size=100, generations=1000, mutation_rate=0.01)

# Visualization or other use of best_path
plt.figure(figsize=(10, 6))
plt.imshow(grid_map_img, cmap='gray') 
for i in range(len(best_path) - 1):
    # Since 'locations' contains coordinates as lists, you need to convert them to tuples if not already done
    start_location = tuple(locations[best_path[i]])
    end_location = tuple(locations[best_path[i+1]])
    
    # Extract the coordinates for plotting
    x_values = [start_location[1], end_location[1]]
    y_values = [start_location[0], end_location[0]]
    
    plt.plot(x_values, y_values, 'ro-')  # Red color for the path
    plt.text(start_location[1], start_location[0], best_path[i], color='white')

# Highlight the start and end positions
start_location = tuple(locations[best_path[0]])
end_location = tuple(locations[best_path[-1]])
plt.plot([start_location[1], end_location[1]],  # 注意这里使用了 [1] 和 [0] 来转置
             [start_location[0], end_location[0]], 'r-')


plt.show()