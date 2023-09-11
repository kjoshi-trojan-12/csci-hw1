#Imports
from queue import PriorityQueue

#Global Variables
NUM_CITIES = 0 #Number of cities


#a* search algorithm
#Parameters: graph, start node, goal node
#Return: a list of nodes that represent the path from the start node to the goal node
#For each node in the graph, the value is the g(n) [0] and f(n) [1] values
def a_star_search(graph, start, goal):
    open = {} #visited
    closed = {} #unvisited

    for node in graph:
        open[node] = (float('inf'), float('inf'), None)

    #initialize start node
    open[start] = (0, heuristic(start, goal), None)
    finished = False
    while not finished:
        if len(closed) == 0:
            finished = True
        else:
            current = min(open, key = open.get) #get the node with the lowest f(n) value

            if current == goal:
                finished = True
                open[current] = closed[current]
            else:
                neighbors = graph[current] #get the neighbors of the current node

                for node in neighbors:
                    if not(open.__contains__(node)):
                        new_g = closed[current][0] + neighbors[node]
                        if new_g < closed[node][0]:
                            closed[node][0] = new_g
                            closed[node][1] = new_g + heuristic(node, goal)
                            closed[node][2] = current
                open[current] = closed[current]
                closed.pop(current)
    return open                


    pass

#Initialize population: initializes the population with a given size and a list of cities
#Return a list of lists, where each list represents a path of size = size
def intialize_population(size: int, cities: list):
    initial_population = []
    for i in range(size):
        initial_population.append(cities)
    return initial_population


#Parent Selection: creates a mating pool of size = size by randomly selecting two parents from the population,
#using roulette wheel selection
#Parameters: population, rank list
#Return: list of populations selected for mating (list contains paths)
def parent_selection(population: list, rank_list: list):
    mating_pool = []
    for i in range(len(population)):
        mating_pool.append(population[rank_list[i]])
    return mating_pool

#Crossover: 

#Read a file:
#Parameters: file path
#Return a list of cities
def read_file_to_city_list(file_path: str):
    cities = []
    counter = 0
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            line = line.split()
            if counter == 0:
                NUM_CITIES = line[0]
            else:
                cities.append((line[0],line[1], line[2]))
            counter += 1
    return cities

def main():
    
    cities = read_file_to_city_list("input.txt")
    intialize_population()
