import warnings
warnings.filterwarnings("ignore")

import grape
import algorithms

import copy
import random

import numpy

from functools import partial



from deap import gp

def progn(*args):
    for arg in args:
        arg()

def prog2(out1, out2):
    return partial(progn,out1,out2)

def prog3(out1, out2, out3):
    return partial(progn,out1,out2,out3)

def if_then_else(condition, out1, out2):
    out1() if condition() else out2()

# class AntSimulator(object):
#     direction = ["north","east","south","west"]
#     dir_row = [1, 0, -1, 0]
#     dir_col = [0, 1, 0, -1]
#
#     def __init__(self, max_moves):
#         self.max_moves = max_moves
#         self.moves = 0
#         self.eaten = 0
#         self.routine = None
#
#     def _reset(self):
#         self.row = self.row_start
#         self.col = self.col_start
#         self.dir = 1
#         self.moves = 0
#         self.eaten = 0
#         self.matrix_exc = copy.deepcopy(self.matrix)
#
#     @property
#     def position(self):
#         return (self.row, self.col, self.direction[self.dir])
#
#     def turn_left(self):
#         if self.moves < self.max_moves:
#             self.moves += 1
#             self.dir = (self.dir - 1) % 4
#
#     def turn_right(self):
#         if self.moves < self.max_moves:
#             self.moves += 1
#             self.dir = (self.dir + 1) % 4
#
#     def move_forward(self):
#         if self.moves < self.max_moves:
#             self.moves += 1
#             self.row = (self.row + self.dir_row[self.dir]) % self.matrix_row
#             self.col = (self.col + self.dir_col[self.dir]) % self.matrix_col
#             if self.matrix_exc[self.row][self.col] == "food":
#                 self.eaten += 1
#             self.matrix_exc[self.row][self.col] = "passed"
#
#     def sense_food(self):
#         ahead_row = (self.row + self.dir_row[self.dir]) % self.matrix_row
#         ahead_col = (self.col + self.dir_col[self.dir]) % self.matrix_col
#         return self.matrix_exc[ahead_row][ahead_col] == "food"
#
#     def if_food_ahead(self, out1, out2):
#         return partial(if_then_else, self.sense_food, out1, out2)
#
#     def run(self,routine):
#         self._reset()
#         while self.moves < self.max_moves:
#             routine()
#
#     def parse_matrix(self, matrix):
#         self.matrix = list()
#         for i, line in enumerate(matrix):
#             self.matrix.append(list())
#             for j, col in enumerate(line):
#                 if col == "#":
#                     self.matrix[-1].append("food")
#                 elif col == ".":
#                     self.matrix[-1].append("empty")
#                 elif col == "S":
#                     self.matrix[-1].append("empty")
#                     self.row_start = self.row = i
#                     self.col_start = self.col = j
#                     self.dir = 1
#         self.matrix_row = len(self.matrix)
#         self.matrix_col = len(self.matrix[0])
#         self.matrix_exc = copy.deepcopy(self.matrix)

class AntSimulator(object):
    direction = ["north", "east", "south", "west"]
    dir_row = [1, 0, -1, 0]
    dir_col = [0, 1, 0, -1]

    def __init__(self, max_moves):
        self.max_moves = max_moves
        self.moves = 0
        self.eaten = 0
        self.routine = None
        self.food_move = 0

    def _reset(self):
        self.row = self.row_start
        self.col = self.col_start
        self.dir = 1
        self.moves = 0
        self.eaten = 0
        self.matrix_exc = copy.deepcopy(self.matrix)
        self.food_move = 0

    def count_food(self):
        return sum(row.count("food") for row in self.matrix)

    @property
    def position(self):
        return (self.row, self.col, self.direction[self.dir])

    def turn_left(self):
        if self.moves < self.max_moves:
            if self.eaten < self.total_food:
                self.food_move += 1
            self.moves += 1
            self.dir = (self.dir - 1) % 4

    def turn_right(self):
        if self.moves < self.max_moves:
            if self.eaten < self.total_food:
                self.food_move += 1
            self.moves += 1
            self.dir = (self.dir + 1) % 4

    def move_forward(self):
        if self.moves < self.max_moves:
            if self.eaten < self.total_food:
                self.food_move += 1
            self.moves += 1
            self.row = (self.row + self.dir_row[self.dir]) % self.matrix_row
            self.col = (self.col + self.dir_col[self.dir]) % self.matrix_col
            if self.matrix_exc[self.row][self.col] == "food":
                self.eaten += 1
            self.matrix_exc[self.row][self.col] = "passed"

    def sense_food(self):
        ahead_row = (self.row + self.dir_row[self.dir]) % self.matrix_row
        ahead_col = (self.col + self.dir_col[self.dir]) % self.matrix_col
        return self.matrix_exc[ahead_row][ahead_col] == "food"

    def if_food_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_food, out1, out2)

    def run(self, routine):
        self._reset()
        while self.moves < self.max_moves:
            routine()

    def parse_matrix(self, matrix):
        self.matrix = list()
        for i, line in enumerate(matrix):
            self.matrix.append(list())
            for j, col in enumerate(line):
                if col == "#":
                    self.matrix[-1].append("food")
                elif col == ".":
                    self.matrix[-1].append("empty")
                elif col == "S":
                    self.matrix[-1].append("empty")
                    self.row_start = self.row = i
                    self.col_start = self.col = j
                    self.dir = 1
        self.matrix_row = len(self.matrix)
        self.matrix_col = len(self.matrix[0])
        self.matrix_exc = copy.deepcopy(self.matrix)
        self.total_food = self.count_food()

ant = AntSimulator(600)

with  open("santafe_trail.txt") as trail_file:
      ant.parse_matrix(trail_file)


pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(ant.if_food_ahead, 2)
pset.addPrimitive(prog2, 2)
pset.addPrimitive(prog3, 3)
pset.addTerminal(ant.move_forward)
pset.addTerminal(ant.turn_left)
pset.addTerminal(ant.turn_right)




from deap import creator, base, tools

import random
import matplotlib.pyplot as plt

problem = 'sensible_ant_problem'

import numpy as np

GRAMMAR_FILE = "BNF_2.bnf"
BNF_GRAMMAR = grape.Grammar(r"grammar/" + GRAMMAR_FILE)


def fitness_eval(individual, points, pset, ant):
    ant._reset()

    if individual.invalid == True:
        return 0,
    else:

        # points = [X, Y]
        # print(individual.phenotype)
        routine = gp.compile(individual.phenotype, pset)
        # print(routine)
        # Run the generated routine
        ant.run(routine)
        return ant.eaten,
        # return ant.eaten + (600 - ant.food_move)/600,


toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
# toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
# toolbox.register("populationCreator", grape.PI_Grow, creator.Individual)


toolbox.register("evaluate", fitness_eval, pset=pset, ant=ant)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=10)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 200
MAX_GENERATIONS = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.01
ELITE_SIZE = 3  # round(0.01*POPULATION_SIZE) #it should be smaller or equal to HALLOFFAME_SIZE
HALLOFFAME_SIZE = 3  # round(0.01*POPULATION_SIZE) #it should be at least 1

MIN_INIT_GENOME_LENGTH = 30  # used only for random initialisation
MAX_INIT_GENOME_LENGTH = 50
random_initilisation = False  # put True if you use random initialisation

MAX_INIT_TREE_DEPTH = 10 # equivalent to 6 in GP with this grammar
MIN_INIT_TREE_DEPTH = 7

MAX_TREE_DEPTH = 20  # equivalent to 17 in GP with this grammar
MAX_WRAPS = 5
CODON_SIZE = 255

CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'list'
MAX_GENOME_LENGTH = None

REPORT_ITEMS = ['gen', 'invalid', 'avg', 'std', 'min', 'max',
                'fitness_test',
                'best_ind_length', 'avg_length',
                'best_ind_nodes', 'avg_nodes',
                'best_ind_depth', 'avg_depth',
                'avg_used_codons', 'best_ind_used_codons',
                #  'behavioural_diversity',
                'structural_diversity',  # 'fitness_diversity',
                'selection_time', 'generation_time']

N_RUNS = 30

move_count = []
best_phenotype_each_run = []

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i)
    print()

    RANDOM_SEED = i


    np.random.seed(RANDOM_SEED)

    print(BNF_GRAMMAR)  # We set up this inside the loop for the case in which the data is defined randomly

    random.seed(RANDOM_SEED)

    # create initial population (generation 0):
    print("creating population")
    if random_initilisation:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                               bnf_grammar=BNF_GRAMMAR,
                                               min_init_genome_length=MIN_INIT_GENOME_LENGTH,
                                               max_init_genome_length=MAX_INIT_GENOME_LENGTH,
                                               max_init_depth=MAX_TREE_DEPTH,
                                               codon_size=CODON_SIZE,
                                               codon_consumption=CODON_CONSUMPTION,
                                               genome_representation=GENOME_REPRESENTATION
                                               )
    else:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE,
                                               bnf_grammar=BNF_GRAMMAR,
                                               min_init_depth=MIN_INIT_TREE_DEPTH,
                                               max_init_depth=MAX_INIT_TREE_DEPTH,
                                               codon_size=CODON_SIZE,
                                               codon_consumption=CODON_CONSUMPTION,
                                               genome_representation=GENOME_REPRESENTATION
                                               )

    # define the hall-of-fame object:
    print("population created")
    hof = tools.HallOfFame(HALLOFFAME_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                            ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                                            bnf_grammar=BNF_GRAMMAR,
                                                            codon_size=CODON_SIZE,
                                                            max_tree_depth=MAX_TREE_DEPTH,
                                                            max_genome_length=MAX_GENOME_LENGTH,
                                                            points_train=None,
                                                            points_test=None,
                                                            codon_consumption=CODON_CONSUMPTION,
                                                            report_items=REPORT_ITEMS,
                                                            genome_representation=GENOME_REPRESENTATION,
                                                            stats=stats, halloffame=hof, verbose=False)

    import textwrap

    best = hof.items[0].phenotype
    best_phenotype_each_run.append(best)
    print("Best individual: \n", "\n".join(textwrap.wrap(best, 80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])

    
    ant._reset()

    with  open("santafe_trail.txt") as trail_file:
        ant.parse_matrix(trail_file)

    routine = gp.compile(hof.items[0].phenotype, pset)
    # print(routine)
    # Run the generated routine
    ant.run(routine)
    move_count.append(ant.food_move)
    ant._reset()

    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(f'Used portion of the genome: {hof.items[0].used_codons / len(hof.items[0].genome):.2f}')

    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    best_ind_length = logbook.select("best_ind_length")
    avg_length = logbook.select("avg_length")

    selection_time = logbook.select("selection_time")
    generation_time = logbook.select("generation_time")
    gen, invalid = logbook.select("gen", "invalid")
    avg_used_codons = logbook.select("avg_used_codons")
    best_ind_used_codons = logbook.select("best_ind_used_codons")

    fitness_test = logbook.select("fitness_test")

    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    structural_diversity = logbook.select("structural_diversity")

    import csv

    r = RANDOM_SEED

    header = REPORT_ITEMS

    with open(r"./results/GE/" + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value],
                             fitness_test[value],
                             best_ind_length[value],
                             avg_length[value],
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             avg_used_codons[value],
                             best_ind_used_codons[value],
                             #  behavioural_diversity[value],
                             structural_diversity[value],
                             #   fitness_diversity[value],
                             selection_time[value],
                             generation_time[value]])
            
import pickle

print(move_count)
with open('./results/GE/max_moves.pkl', 'wb') as fp:
    pickle.dump(move_count, fp)

with open('./results/GE/best_phenotype_each_run.pkl', 'wb') as fp:
    pickle.dump(best_phenotype_each_run, fp)

# [600, 600, 524, 580, 600, 574, 542, 600, 600, 530, 588, 600, 600, 600, 600, 574, 600, 600, 566, 594, 574, 570, 600, 580, 568, 600, 600, 580, 600, 600]

# [600, 584, 584, 584, 382, 490, 600, 600, 600, 522, 513, 534, 588, 588, 600, 380, 600, 506, 570, 600, 600, 586, 402, 542, 414, 536, 600, 586, 600, 600]
