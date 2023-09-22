#Imports
from __future__ import annotations
import math
from queue import PriorityQueue
import queue
from operator import attrgetter
import random
from itertools import permutations
import sys
import numpy as np
import pandas as pd


#Global Variables
num_cities = 0 #Number of cities
rank_list = [] #List of tuples that contain the index [0] and fitness value [1] of each individual
population = [] #List of paths (a permutation of cities)
INPUT_FILE = "input_2.txt"
cities = [] #List of cities
start_location = None #Starting location (city)
end_location = None #Ending location (city)
#Hyperparameters
pop_size = 100 #Population size
MAX_GENERATIONS = 30 #Maximum number of generations to run the algorithm for
mutation_rate = 0.01 #Mutation rate
elite_size = 30 #Number of elite individuals to keep in the population

#Node data structure represented by four components:
#State of the node
#Parent of the node
#Action that was taken to get to the node
#Path cost of the node g(n) or g(node)
class Node:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
        # self.path_cost = path_cost


    #Function to check if the node is the goal node
    #Parameters: node, goal node
    #Return: True if the node is the goal node, False otherwise
    def is_goal(self):
        return self.state == "Home" and self.parent != None
    
    #Get path cost: gets the path cost of the node
    #Return: path cost (float)
    # def get_path_cost(self):
    #     return self.path_cost
    
    def get_state(self):
        return self.state
    
    def __repr__(self) -> str:
        return f"State: {self.state}, Action: {self.action}"
    

    


#Class for Fitness: used to compare the fitness of two individuals
class Fitness:
    def __init__(self, path) -> None:
        self.path = path
        self.distance = 0
        self.fitness = 0.0

    #Function to calculate the distance of the path
    #Return: distance (float)
    def path_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(len(self.path)):
                from_city = self.path[i]
                to_city = None
                if i + 1 < len(self.path):
                    to_city = self.path[i+1]
                else:
                    to_city = self.path[0]
                path_distance += from_city.calculate_distance(to_city)
            self.distance = path_distance
        return self.distance
    
    #Function to calculate the fitness of the path
    #Return: fitness (float)
    def path_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.path_distance())
        return self.fitness

#City data structure is represented as a node (specified above)
#and also contains coordinates for the location (x, y, z)
class City(Node):
    def __init__(self, state, parent, action, x, y, z):
        #state is the name of the city (string)
        #parent is the parent node (or in this case, the previous city)
        #action is the action taken to get to the city (string)
        #path_cost is the cost to get to the city (float)
        #x, y, z are the coordinates of the city (int)
        super().__init__(state, parent, action)
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return super().__repr__() + f", Coordinates: ({self.x}, {self.y}, {self.z})"



    #Calculate Distace: calculates the distance between two cities
    #Parameters: city1, city2
    #Return: distance between city1 and city2
    def calculate_distance(self, other_city):
        return math.sqrt((self.x - other_city.x)**2 + (self.y - other_city.y)**2 + (self.z - other_city.z)**2)
            

#Probablity generator: generates a random number between 0 and 1
#Return: a random number between 0 and 1
def probability_generate():
    return random.random()

#Genetic Search Algorithm
#Parameters: population
#Return: the best solution found by the algorithm 
def genetic_search():
    global population
    generations = MAX_GENERATIONS
    for i in range(generations):
        population_ranked = generate_rank_list(population)
        mating_pool = create_mating_pool(population_ranked, elite_size)     
        
        #Crossover
        children = []
        pool = random.sample(mating_pool, len(mating_pool))
        for i in range(0, elite_size):
            children.append(mating_pool[i])
        for i in range(0, len(mating_pool) - elite_size):
            child = crossover(pool[i], pool[len(mating_pool) - i - 1])
            children.append(child)
        #Mutation
        new_population = []
        for i in range(len(children)):
            mutate_index = mutate(children[i], mutation_rate)
            new_population.append(mutate_index)
        population = new_population
    global rank_list
    #Generate the rank list for the final population and ensure that the best individual is returned with 
    # the best fitness value and the last city is the end location with the coordinates (0, 0, 0)
    rank_list = generate_rank_list(population)
    return get_best_individual(population, rank_list)
    

            


#Get best individual: gets the best individual from the population
#Parameters: population, rank list
#Return: the best individual
def get_best_individual(population: list, rank_list: list):
    return population[rank_list[0][0]]



#Initialize population: initializes the population with a given size and a list of cities
#Return a list of paths (a permutation of cities) of size = size
def initialize_population(cities: list, size: int = pop_size):
    population = []
    for i in range(size):
        population.append(random.sample(cities, len(cities)))
    return population

#Mating pool selection: creates a mating pool of size = size by randomly selecting two parents from the population,
def create_mating_pool(population_ranked: list, elite_size: int):
    mating_pool = []
    #population_ranked is a list of tuples that contain the index [0] and fitness object [1] of each individual
    #create a list of tuples that contain the index and fitness value of each individual
    population_ranked = [(i, population_ranked[i][1].fitness) for i in range(len(population_ranked))]
    #select the elite individuals for the mating pool
    selection = []
    selection_df = pd.DataFrame(np.array(population_ranked), columns = ['Index', 'Fitness'])
    selection_df['cum_sum'] = selection_df.Fitness.cumsum()
    selection_df['cum_perc'] = 100 * selection_df.cum_sum/selection_df.Fitness.sum()

    for i in range(elite_size):
        selection.append(population_ranked[i][0])
    for i in range(len(population_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(len(population_ranked)):
            if pick <= selection_df.iat[i, 3]:
                selection.append(population_ranked[i][0])
                break
    
    for i in range(len(selection)):
        index = selection[i]
        mating_pool.append(population[index])
    return mating_pool

#Crossover function: creates a child from two parents
#Parameters: parent1, parent2, start_index, end_index
#Return: child (list of cities)
def crossover(parent1, parent2):
    child_chromosome = []
    child_chromosome_p1 = []
    child_chromosome_p2 = []

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_chromosome_p1.append(parent1[i])
    for i in range(len(parent2)):
        if parent2[i] not in child_chromosome_p1:
            child_chromosome_p2.append(parent2[i])
    
    child_chromosome = child_chromosome_p1 + child_chromosome_p2
    return child_chromosome

def mutate(individual, mutation_rate: float):
    for swapped in range(len(individual)):
        mutation_probability = probability_generate()
        if(mutation_probability < mutation_rate):
            swap_with = int(random.random() * len(individual))
            chromosome_1 = individual[swapped]
            chromosome_2 = individual[swap_with]
            individual[swapped] = chromosome_2
            individual[swap_with] = chromosome_1
    return individual




    


#Read a file with the first line being the number of cities
#next n lines are the x, y, z coordinates of each city (separated by a space)
#Return a list of cities
def read_file_to_city_list(file_name: str):
    cities = []
    # cities.append(start_location) #add the start location to the list of cities
    with open(file_name, "r") as file:
        num_cities = int(file.readline())
        for i in range(num_cities):
            line = file.readline()
            line = line.split()
            x, y, z = int(line[0]), int(line[1]), int(line[2])
            city = City("City " + str(i), None, "Walk to", x, y, z)
            cities.append(city)
    cities.append(end_location) #add the end location to the list of cities
    return cities

#Generate rank list: generates a list of tuples that contain the index [0] and 
# fitness value [1] of each individual sorted in descending order
#Parameters: population
#Return: rank list
def generate_rank_list(population: list):
    fitness_calculations = {}
    for i in range(len(population)):
        cur_fitness_calculation = Fitness(population[i])
        cur_fitness_calculation.path_fitness()
        fitness_calculations[i] = cur_fitness_calculation
    
    ranked_fitness_calculations = sorted(fitness_calculations.items(), key = lambda calculation: calculation[1].fitness , reverse = True)
    return ranked_fitness_calculations

#Function to write output to a file (output.txt) with the following format:
#1st line: computed distance of the path
#Next n+1 lines: each line has three non-negative integers separated by a space (coordinates of the city)
#indicating the city visited in order
#n+1 lines because we need to return to the starting city
#path = population
#Parameters: path
def write_output_to_file(path: list):
    with open("output.txt", "w") as file:
        file.write(str(rank_list[0][1].distance) + "\n")
        for city in path:
            file.write(str(city.x) + " " + str(city.y) + " " + str(city.z) + "\n")


def main():    
    global start_location, end_location, cities, population, num_cities, rank_list
    # start_location = City("Home", None, "Walk out of the house", 0, 0, 0)
    end_location = City("Home", None, "Walk back into the house", 0, 0, 0)
    cities = read_file_to_city_list(INPUT_FILE)
    num_cities = len(cities)
    

    print(f"Number of cities: {num_cities}\nList of cities\n: {cities}")
    print("Initializing population...")
    population = initialize_population(cities, pop_size)
    print("Population initialized!")
    print("Running genetic search algorithm...")
    best_path = genetic_search()
    print("Genetic search algorithm completed!")
    print("Best path:\n",best_path,"\nDistance:", rank_list[0][1].distance)
    write_output_to_file(best_path)

if __name__ == "__main__":
    main()




