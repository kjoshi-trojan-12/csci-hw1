#Imports
from queue import PriorityQueue
import queue
from operator import attrgetter



#Global Variables
NUM_CITIES = 0 #Number of cities
OPTIMAL_FITNESS = 0 #Optimal fitness value (fine-tune this)



#Node data structure represented by four components:
#State of the node
#Parent of the node
#Action that was taken to get to the node
#Path cost of the node g(n) or g(node)
#Heuristic value of the node h(n) or h(node)
#Estimated cost of the best solution through the node f(n) or f(node)
class Node:
    def __init__(self, state, parent, action, path_cost, heuristic = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.f = self.path_cost + self.heuristic


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()

    def mutated_genes(self):
        global GENES
        gene = random.choice(GENES)
        return gene
#City data structure is represented as a node (specified above)
#and also contains coordinates for the location (x, y, z)
class City(Node, Individual):
    def __init__(self, state, parent, action, path_cost, x, y, z):
        super().__init__(state, parent, action, path_cost, heuristic = 0, chromosome = None)
        #state is the name of the city (string)
        #parent is the parent node (or in this case, the previous city)
        #action is the action taken to get to the city (string)
        #path_cost is the cost to get to the city (float)
        self.x = x
        self.y = y
        self.z = z




#a* search algorithm
#Parameters: graph, start node, goal node
#Return: a list of nodes that represent the path from the start node to the goal node
#For each node in the graph, the value is the g(n) [0] and f(n) [1] values
# def a_star_search(graph, start, goal):
#     setattr(start, 'path_cost', 0) #set the path cost of the start node to 0
#     open = [start] #visited
#     closed = [] #unvisited

#     while(len(open) != 0):
#         #Find the node with the lowest f(n) value
#         q = min(open, key = attrgetter('path_cost')) #get the node with the lowest f(n) value
#         open.remove(q)

#         #Generate the successors of q and set their parents to q
#         successors = graph[q]
#         for node in successors:
#             if node == goal:
#                 return node
#             #else if node with the same state as successor is in open 
#             # with lower f value
#             #skip this successor

#             elif node.state
#             else:
#                 #Compute the g(n) and h(n) values of the successors
#                 new_g = q.path_cost + 0 #TO-DO: Distance between q and node
#                 setattr(node, 'path_cost', new_g)
#                 new_h = 0 #TO-DO: Distance between node and goal
#                 setattr(node, 'heuristic', new_h)
#                 setattr(node, 'f', new_g + new_h)
            

#Genetic Search Algorithm
#Parameters: population, fitness function
#Return: the best solution found by the algorithm 
def genetic_search(population: list, fitness_function):
    #repeat until some individual is fit enough or enough time has elapsed in the 
    # population
    while True:
        #weights
        population2 = []
        for i in range(len(population)):
            parent1 = None #To-Do: Weighted random selection (population, weights, 2)
            parent2 = None #To-Do: Weighted random selection (population, weights, 2)
            child = crossover(parent1, parent2) #To-Do: Crossover
            if True: #To-do: Small random probability
                child = mutate(child) #To-Do: Mutation
            population2.append(child)
            population = population2
    return max(population, key = fitness_function) #To-Do: Fitness function


#Initialize population: initializes the population with a given size and a list of cities
#Return a list of lists, where each list represents a path of size = size
def intialize_population(size: int, cities: list):
    initial_population = []
    for i in range(size):
        initial_population.append(cities)
    return initial_population


#Parent Selection: creates a mating pool of size = size by randomly selecting two parents from the population,
#using roulette wheel selection
#Parameters: population, rank list (list of tuples that contain the index [0] and fitness value [1] of each individual)
#Return: list of populations selected for mating (list contains paths)
def parent_selection(population: list, rank_list: list):
    mating_pool = []
    for i in range(len(population)):
        mating_pool.append(population[rank_list[i]])
    return mating_pool

#Crossover function: creates a child from two parents
#Parameters: parent1, parent2, start_index, end_index
#Return: child (list of cities)
def crossover(parent1: list, parent2: list, start_index: int, end_index: int):
    child = []
    num_chromosomes = len(parent1)
    for i in range(num_chromosomes):
        if i < start_index or i > end_index:
            child.append(parent2[i])
        else:
            child.append(parent1[i])
    return child

#Mutation function: mutates a child


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
