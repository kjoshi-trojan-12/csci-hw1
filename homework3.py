#Imports
from __future__ import annotations
import math
from queue import PriorityQueue
import queue
from operator import attrgetter
import random
from itertools import permutations
import sys


#Global Variables
num_cities = 0 #Number of cities
OPTIMAL_FITNESS = 300 #Optimal fitness value (fine-tune this)
rank_list = [] #List of tuples that contain the index [0] and fitness value [1] of each individual
global population #List of paths (a permutation of cities)
INPUT_FILE = "input_1.txt"
num_cities_visited = 0 #Number of cities visited
cities = [] #List of cities
start_location = None #Starting location (city)
end_location = None #Ending location (city)
search_iterations = 0 #Number of iterations of the search algorithm
pop_size = 15 #Population size
connected_matrix = [] #Connection matrix
cost_matrix = [] #Cost matrix
MAX_GENERATIONS = 5 #Maximum number of generations to run the algorithm for



#Node data structure represented by four components:
#State of the node
#Parent of the node
#Action that was taken to get to the node
#Path cost of the node g(n) or g(node)
class Node:
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost


    #Function to check if the node is the goal node
    #Parameters: node, goal node
    #Return: True if the node is the goal node, False otherwise
    def is_goal(self):
        return self.state == "Home" and self.parent != None
    
    #Get path cost: gets the path cost of the node
    #Return: path cost (float)
    def get_path_cost(self):
        return self.path_cost
    
    def get_state(self):
        return self.state
    
    def __repr__(self) -> str:
        return f"State: {self.state}, Action: {self.action}, Path Cost: {self.path_cost}"
    
    #Function to expand the node
    #Return: list of nodes that are the successors of the node
    def expand(self):
        successors = []
        for city in cities:
            if city.state != self.state:
                new_path_cost = self.path_cost + calculate_distance(self, city) # type: ignore
                new_node = Node(city.state, self, None, new_path_cost)
                successors.append(new_node)
        return successors
    

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome #sequence of genes (objects)
        self.num_genes = len(chromosome)
        self.fitness = self.calculate_fitness()

    def mutate(self):
        #swap two cities in the chromosome
        index1 = random.randint(1, self.num_genes - 2)
        index2 = random.randint(1, self.num_genes - 2)
        while index1 == index2:
            index2 = random.randint(0, self.num_genes - 1)
        temp = self.chromosome[index1]
        self.chromosome[index1] = self.chromosome[index2]
        self.chromosome[index2] = temp

    
    def __repr__(self) -> str:
        return f"Chromosome: {self.chromosome}, Fitness: {self.fitness}"
    
    
    
    # def __le__(self, other):
    #     return self.fitness >= other.fitness
    # def __ge__(self, other):
    #     return self.fitness <= other.fitness

    
    #Function to calculate the fitness of the individual
    #Return: fitness value (float)
    def calculate_fitness(self):
        fitness = 0
        for i in range(self.num_genes - 1):
            fitness += calculate_distance(self.chromosome[i], self.chromosome[i+1])
        return fitness
    
    #Function to get the path of the individual
    #Return: list of cities
    def get_path(self):
        return self.chromosome
    
    #Function to get the path cost of the individual
    #Return: path cost (float)
    def get_path_cost(self):
        return self.fitness
    
    #Function to get the number of cities visited
    #Return: number of cities visited (int)
    def get_num_cities_visited(self):
        return self.num_genes - 2
    
    #Function to get the number of cities in the path
    #Return: number of cities in the path (int)
    def get_num_cities_in_path(self):
        return self.num_genes
    
    #Function to get the fitness of the individual
    #Return: fitness value (float)
    def get_fitness(self):
        return self.fitness
    
    #Function to get the chromosome of the individual
    #Return: chromosome (list of cities)
    def get_chromosome(self):
        return self.chromosome
    
    #Function to set the chromosome of the individual
    #Parameters: chromosome (list of cities)
    def set_chromosome(self, chromosome):
        self.chromosome = chromosome
        self.num_genes = len(chromosome)
        self.fitness = self.calculate_fitness()

    #Function to set the fitness of the individual
    #Parameters: fitness value (float)
    def set_fitness(self, fitness):
        self.fitness = fitness

    #Function to set the number of cities visited
    #Parameters: number of cities visited (int)
    def set_num_cities_visited(self, num_cities_visited):
        self.num_cities_visited = num_cities_visited

    #Function to set the number of cities in the path
    #Parameters: number of cities in the path (int)
    def set_num_cities_in_path(self, num_cities_in_path):
        self.num_cities_in_path = num_cities_in_path

    #Function to set the path cost of the individual
    #Parameters: path cost (float)
    def set_path_cost(self, path_cost):
        self.path_cost = path_cost
    


#City data structure is represented as a node (specified above)
#and also contains coordinates for the location (x, y, z)
class City(Node):
    def __init__(self, state, parent, action, path_cost, x, y, z):
        #state is the name of the city (string)
        #parent is the parent node (or in this case, the previous city)
        #action is the action taken to get to the city (string)
        #path_cost is the cost to get to the city (float)
        #x, y, z are the coordinates of the city (int)
        super().__init__(state, parent, action, path_cost)
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return super().__repr__() + f", Coordinates: ({self.x}, {self.y}, {self.z})"
        #return f"City: {self.state}, Coordinates: ({self.x}, {self.y}, {self.z})"



#Calculate Distace: calculates the distance between two cities
#Parameters: city1, city2
#Return: distance between city1 and city2
def calculate_distance(city1: City, city2: City):
    return math.dist((city1.x, city1.y, city1.z), (city2.x, city2.y, city2.z))
    # return math.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2 + (city1.z - city2.z)**2)


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
#Parameters: population
#Return: the best solution found by the algorithm 
def genetic_search(population: list):
    #repeat until some individual is fit enough or enough time has elapsed in the 
    # population
    #Set the destination city as the goal node
    gen = 1
    while gen <= MAX_GENERATIONS:
        print("Generation: ", gen)
        p = 1
        while(p <= pop_size):
            print("Population: ", p)
            #Selection
            #Select two parents from the population
            #random crossover point
            start_index = random.randint(1, num_cities - 2)
            end_index = random.randint(1, num_cities - 2)
            while start_index == end_index: #make sure start_index != end_index
                end_index = random.randint(1, num_cities - 2)
            if start_index > end_index: #make sure start_index < end_index, swap if not
                temp = start_index
                start_index = end_index
                end_index = temp
            parents = parent_selection(population)
            parent1, parent2 = parents[0], parents[1]
            if parent1 == None or parent2 == None:
                print("Parent is None")
                sys.exit()
            print("Parents chosen:\n", parent1, "\n", parent2)
            child = crossover(parent1, parent2, start_index, end_index)
            #Mutation
            mutation_probability = probability_generate()
            if(mutation_probability <= 0.02): 
                print("Mutation occurred! with ", mutation_probability)
                child.mutate()
            #Compute total cost of the candidate path using the objective function
            child.fitness = objective_function(child.chromosome)
            population.append(child)
            #Replace the worst individual in the population with the child
            global rank_list
            rank_list = generate_rank_list(population)
            worst_individual_index = rank_list[-1][0]
            population[worst_individual_index] = child
            p += 1
        gen += 1
        if gen > MAX_GENERATIONS:
            break

    #Save the candidate path with the lowest cost between the start and end locations as the best solution
    #Return the best solution using the objective function
    rank_list = generate_rank_list(population)
    best_individual = get_best_individual(population, rank_list)
    return best_individual
            


#Get best individual: gets the best individual from the population
#Parameters: population, rank list
#Return: the best individual
def get_best_individual(population: list, rank_list: list):
    return population[rank_list[0][0]]

#Objective function: cost of the candidate solution path 
#to compare the solutions and determine the best solution
#Cost of candidate path is calculated when it satisfies two conditions:
#1. It is a valid path (chromosome contains at least two none zero elements)
#2 Chromosome contains a connected path (each node/city connects at least one other node/city) 
# with the start location as first node and end location as last node
#Parameters: chromosome
#Return: cost of the candidate solution path
def objective_function(chromosome: list):
    #Check if the chromosome is a valid path
    if not is_valid_path(chromosome):
        return math.inf
    #Check if the chromosome contains a connected path
    if not is_connected_path(chromosome):
        return math.inf
    #If the chromosome is a valid path and contains a connected path, calculate the cost
    cost = 0
    for i in range(len(chromosome) - 1):
        cost += calculate_distance(chromosome[i], chromosome[i+1])
    return cost

#Is valid path: checks if the chromosome is a valid path
#Parameters: chromosome
#Return: True if the chromosome is a valid path, False otherwise
def is_valid_path(chromosome: list):
    #Check if the chromosome contains at least two none zero elements
    if len(chromosome) < 2:
        return False
    return True

#Is connected path: checks if the chromosome contains a connected path
#Parameters: chromosome
#Return: True if the chromosome contains a connected path, False otherwise
def is_connected_path(chromosome: list):
    #Check if the chromosome contains a connected path
    #Check if the first and last elements are the start and end locations
    if chromosome[0].state != start_location.state or chromosome[-1].state != end_location.state:
        return False
    #Check if each node/city connects at least one other node/city
    for i in range(len(chromosome) - 1):
        if connected_matrix[i][i+1] == 0:
            return False
    return True


#Generate connection matrix: generates a connection matrix for the cities
# using binary values (1 if connected, 0 if not connected)
#Parameters: cities
#Return: connection matrix
def generate_connection_matrix(cities: list):
    connection_matrix = []
    for i in range(num_cities):
        row = []
        for j in range(num_cities):
            if i == j:
                row.append(0)
            else:
                row.append(1)
        connection_matrix.append(row)
    return connection_matrix

#Generate cost matrix: generates a cost matrix for the cities
#Parameters: cities
#Return: cost matrix
def generate_cost_matrix(cities: list):
    cost_matrix = []
    for i in range(num_cities):
        row = []
        for j in range(num_cities):
            if i == j:
                row.append(0)
            else:
                row.append(calculate_distance(cities[i], cities[j]))
        cost_matrix.append(row)
    return cost_matrix

#Initialize population: initializes the population with a given size and a list of cities
#Return a list of paths (a permutation of cities) of size = size
def initialize_population(cities: list):
    population = []
    #Generate all permutations of the cities
    permutations_cities = list(permutations(cities))
    #Major Step 1: Randomly generate a chromosome for each permutation
    #Major Step 2: Check if the chromosome represents a valid path
    #If not, skip it
    #Major Step 3: Create an individual from the chromosome
    #Major Step 4: Add the individual to the population
    #Major Step 5: Repeat until the population is full
    list_counter = 0
    while len(population) < pop_size:
        print("Population size: ", len(population))
        permutation = permutations_cities[list_counter]
        #Convert the permutation to a list of cities
        permutation = list(permutation)
        #Major step 2: Check if the permutation represents a valid path
        #If not, skip it
        valid_path = True
        for i in range(len(permutation) - 1):
            if permutation[i].state == permutation[i+1].state:
                valid_path = False
                break
        if not valid_path:
            continue      
        #Create an individual from the permutation
        population.append(Individual(permutation))
        list_counter += 1
    #Write the population to a file
    with open("population.txt", "w") as file:
        for individual in population:
            file.write(str(individual) + "\n")
    return population
    
    # while len(population) < pop_size:
    #     permutation = permutations_cities[list_counter]
    #     #Convert the permutation to a list of cities
    #     permutation = list(permutation)
    #     #Major step 2: Check if the permutation represents a valid path
    #     #If not, skip it
    #     valid_path = True
    #     for i in range(len(permutation) - 1):
    #         if permutation[i].state == permutation[i+1].state:
    #             valid_path = False
    #             break
    #     if not valid_path:
    #         continue      
    #     #Create an individual from the permutation
    #     population.append(Individual(permutation))
    #     list_counter += 1
    # #Write the population to a file
    # with open("population.txt", "w") as file:
    #     for individual in population:
    #         file.write(str(individual) + "\n")
    # return population


#Parent Selection: creates a mating pool of size = size by randomly selecting two parents from the population,
#using roulette wheel selection
#Parameters: population, rank list
#Return: list of parents
def parent_selection(population: list):
    #Select two parents from the population
    parents = []
    parents = rank_selection(population, 2)
    return parents

#Heuristic path generator: generate a path from the current city to the goal city
#Parameters: current city, goal city
#Return: a list of cities that represent the path from the current city to the goal city


#Crossover function: creates a child from two parents
#Parameters: parent1, parent2, start_index, end_index
#Return: child (list of cities)
def crossover(parent1, parent2, start_index: int, end_index: int):
    child_chromosome = []
    for i in range(parent1.num_genes):
        if i < start_index or i > end_index:
            child_chromosome.append(parent2.chromosome[i])
        else:
            child_chromosome.append(parent1.chromosome[i])
    return Individual(child_chromosome)




#Weighted random selection: selects a random individual from the population based on their fitness
#Parameters: population, weights, number of individuals to select
#Return: list of selected individuals
def weighted_random_selection(population: list, weights: list, num_individuals: int):
    selected_individuals = []
    for i in range(num_individuals):
        selected_individuals.append(random.choices(population, weights, k = 1))
    return selected_individuals

#Roulette wheel selection: 
#Steps:
#1 calculate the sum of the fitness values of all individuals in the population (s)
#2 generate a random number between 0 and s
#3 starting from the top of the popultion
# a. keep adding the fitness values to the partial sum (p), until p < s
#4 the individual which p exceeds s is the selected individual
#Parameters: population, number of individuals to select
#Return: list of selected individuals
def roulette_wheel_selection(population: list):
    #Calculate the sum of the fitness values of all individuals in the population
    s = 0
    for individual in population:
        s += individual.fitness
    #Generate a random number between 0 and s
    random_number = random.uniform(0, s)
    #Starting from the top of the population
    #Keep adding the fitness values to the partial sum (p), until p < s
    #The individual which p exceeds s is the selected individual
    p = 0
    for individual in population:
        p += individual.fitness
        if p > random_number:
            return individual
        

#Rank selection: selects a random individual from the population based on their rank
#Parameters: population, number of individuals to select
#Return: list of selected individuals
def rank_selection(population: list, num_individuals: int):
    selected_individuals = []
    rank_list_weights = []
    for indivdual in rank_list:
        if indivdual[1] == 0:
            rank_list_weights.append(1)
        elif indivdual[1] == math.inf:
            rank_list_weights.append(0)
        else:
            rank_list_weights.append(indivdual[1])
        
    print("Rank list weights: ", rank_list_weights)
    for i in range(num_individuals):
        chosen_parent = random.choices(population, rank_list_weights, k = 1)
        selected_individuals.append(chosen_parent[0])
    return selected_individuals


#Read a file with the first line being the number of cities
#next n lines are the x, y, z coordinates of each city (separated by a space)
#Return a list of cities
def read_file_to_city_list(file_name: str):
    cities = []
    cities.append(start_location) #add the start location to the list of cities
    with open(file_name, "r") as file:
        num_cities = int(file.readline())
        for i in range(num_cities):
            line = file.readline()
            line = line.split()
            x, y, z = int(line[0]), int(line[1]), int(line[2])
            city = City("City " + str(i), None, "Walk to", 0, x, y, z)
            cities.append(city)
    cities.append(end_location) #add the end location to the list of cities
    return cities

#Generate rank list: generates a list of tuples that contain the index [0] and 
# fitness value [1] of each individual sorted in descending order
#Parameters: population
#Return: rank list
def generate_rank_list(population: list):
    rank_list = []
    for i in range(len(population)):
        rank_list.append((i, population[i].fitness))
    rank_list.sort(key = lambda x: x[1])
    return rank_list

#Function to write output to a file (output.txt) with the following format:
#1st line: computed distance of the path
#Next n+1 lines: each line has three non-negative integers separated by a space
#indicating the city visited in order
#n+1 lines because we need to return to the starting city
#path = population
def write_output_to_file(best_route: Individual):
    with open("output.txt", "w") as file:
        best_route.fitness
        file.write(str(best_route.fitness) + "\n")
        path = best_route.chromosome
        for city in path:
            file.write(str(city.x) + " " + str(city.y) + " " + str(city.z) + "\n")
        file.write("Number of iterations: " + str(search_iterations) + "\n")
        


def main():    
    global start_location, end_location, cities, population, num_cities, rank_list, connected_matrix, cost_matrix
    start_location = City("Home", None, "Walk out of the house", 0, 0, 0, 0)
    end_location = City("Home", None, "Walk back into the house", 0, 0, 0, 0)
    cities = read_file_to_city_list(INPUT_FILE)
    num_cities = len(cities)
    connected_matrix = generate_connection_matrix(cities)
    print("Connected matrix:\n", connected_matrix)
    cost_matrix = generate_cost_matrix(cities)
    

    print(f"Number of cities: {num_cities}\nList of cities\n: {cities}")
    print("Initializing population...")
    population = initialize_population(cities)
    rank_list = generate_rank_list(population)
    print("Population initialized!")
    print("Running genetic search algorithm...")
    best_path = genetic_search(population)
    print("Genetic search algorithm completed!")
    print(best_path)
    write_output_to_file(best_path)
    print("Num of search iterations: ", search_iterations)

if __name__ == "__main__":
    main()




