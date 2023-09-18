#Imports
from __future__ import annotations
import math
from queue import PriorityQueue
import queue
from operator import attrgetter
import random


#Global Variables
NUM_CITIES = 0 #Number of cities
OPTIMAL_FITNESS = 0 #Optimal fitness value (fine-tune this)
RANK_LIST = [] #List of tuples that contain the index [0] and fitness value [1] of each individual
POPULATION = [] #List of paths (a permutation of cities)
MATING_POOL = [] #List of populations selected for mating (list contains paths)



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
        self.chromosome = chromosome #sequence of genes (objects)
        self.num_genes = len(chromosome)
        self.fitness = self.calculate_fitness()

    def mutate(self):
        #swap two cities in the chromosome
        index1 = random.randint(0, self.num_genes - 1)
        index2 = random.randint(0, self.num_genes - 1)
        while index1 == index2:
            index2 = random.randint(0, self.num_genes - 1)
        temp = self.chromosome[index1]
        self.chromosome[index1] = self.chromosome[index2]
        self.chromosome[index2] = temp

    #Fitness function: calculates the fitness of an individual
    #Parameters: individual (list of cities)
    #Return: fitness value (float)
    def calculate_fitness(self):
        fitness = 0
        for i in range(self.num_genes - 1):
            city1 = self.chromosome[i]
            city2 = self.chromosome[i + 1]
            distance = city1.calculate_distance(city2)
            fitness += distance


#City data structure is represented as a node (specified above)
#and also contains coordinates for the location (x, y, z)
class City(Node):
    def __init__(self, state, parent, action, path_cost, x, y, z):
        super().__init__(state, parent, action, path_cost, heuristic = 0, chromosome = None)
        #state is the name of the city (string)
        #parent is the parent node (or in this case, the previous city)
        #action is the action taken to get to the city (string)
        #path_cost is the cost to get to the city (float)
        self.x = x
        self.y = y
        self.z = z

    #Calculate the distance between two cities
    #Parameters: city1, city2
    #Return: distance between city1 and city2
    def calculate_distance(self, other_city: City):
        return math.sqrt((self.x - other_city.x)**2 + (self.y - other_city.y)**2 + (self.z - other_city.z)**2)
        




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
            

#Probablity generator: generates a random number between 0 and 1
#Return: a random number between 0 and 1
def probability_generate():
    return random.random()

#Genetic Search Algorithm
#Parameters: population, fitness function
#Return: the best solution found by the algorithm 
def genetic_search(population: list, fitness_function):
    #repeat until some individual is fit enough or enough time has elapsed in the 
    # population
    while True:
        weighted_by_population_and_fitness(population, 2)
        population2 = []
        for i in range(len(population)):
            parent1 = None #To-Do: Weighted random selection (population, weights, 2)
            parent2 = None #To-Do: Weighted random selection (population, weights, 2)
            child = crossover(parent1, parent2)
            if(probability_generate() <= 0.01): 
                child.mutate()
            population2.append(child)
            population = population2
        if max(population, key = fitness_function).fitness >= OPTIMAL_FITNESS:
            break
    return max(population, key = fitness_function) #To-Do: Fitness function


#Initialize population: initializes the population with a given size and a list of cities
#Return a list of paths (a permutation of cities) of size = size
def initialize_population(size: int, cities: list):
    initial_population = []
    for i in range(size):
        path = random.sample(cities, len(cities))
        individual = Individual(path)
        initial_population.append(individual)
    return initial_population


#Parent Selection: creates a mating pool of size = size by randomly selecting two parents from the population,
#using roulette wheel selection
#Parameters: population, rank list (list of tuples that contain the index [0] and fitness value [1] of each individual)
#Return: list of populations selected for mating (list contains paths)
def parent_selection(population: list, rank_list: list):
    mating_pool = []
    for i in range(len(population)):
        parent1 = roulette_wheel_selection(population, 1)
        parent2 = roulette_wheel_selection(population, 1)
        mating_pool.append(parent1)
        mating_pool.append(parent2)
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

#Weighted random selection: selects a random individual from the population based on their fitness
#Parameters: population, weights, number of individuals to select
#Return: list of selected individuals
def weighted_random_selection(population: list, weights: list, num_individuals: int):
    selected_individuals = []
    for i in range(num_individuals):
        selected_individuals.append(random.choices(population, weights, k = 1))
    return selected_individuals

#weighted by population and fitness
#Parameters: population, number of individuals to select
#Return: list of selected individuals
def weighted_by_population_and_fitness(population: list, num_individuals: int):
    selected_individuals = []
    for i in range(num_individuals):
        selected_individuals.append(random.choices(population, k = 1))
    return selected_individuals

#Roullete wheel selection: selects a random individual from the population based on their fitness
#Parameters: population, number of individuals to select
#Return: list of selected individuals
def roulette_wheel_selection(population: list, num_individuals: int):
    selected_individuals = []
    total_fitness = 0
    for individual in population:
        total_fitness += individual.fitness
    for i in range(num_individuals):
        random_num = random.uniform(0, total_fitness)
        current_sum = 0
        for individual in population:
            current_sum += individual.fitness
            if current_sum > random_num:
                selected_individuals.append(individual)
                break
    return selected_individuals

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
