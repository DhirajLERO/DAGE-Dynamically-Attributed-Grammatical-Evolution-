import operator
import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp
from deap import algorithms

import random

import numpy
import copy

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
        # print(self.eaten, self.food_move)
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



# Convert trees to flat lists of item indices
def fitness_eval(individual):

    # Transform the tree expression to functional Python code
    try:
        routine = gp.compile(individual, pset)
        # Run the generated routine
        ant.run(routine)
        return ant.eaten,
    except Exception as e:
        return 0,

# DEAP Creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=5)




toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", fitness_eval)
toolbox.register("select", tools.selTournament, tournsize=10)


toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

# GP Run



N_RUNS = 30
# collect_best = []

move_count = []
best_phenotype_each_run = []
for i in range(N_RUNS):
    print()
    print()
    print("Run:", i)
    print()

    RANDOM_SEED = i
    np.random.seed(RANDOM_SEED)

    random.seed(RANDOM_SEED)
    pop = toolbox.population(n=200)

    hof = tools.HallOfFame(3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, 0.9, 0.01, 200, stats, halloffame=hof, verbose=True)

# Print best solution
    best_ind = hof[0]

    best_phenotype_each_run.append(best_ind)

    # ant = AntSimulator(600)

    ant._reset()
    with  open("santafe_trail.txt") as trail_file:
        ant.parse_matrix(trail_file)

    routine = gp.compile(best_ind, pset)
    # print(routine)
    # Run the generated routine
    ant.run(routine)
    print(ant.__dict__)
    move_count.append(ant.food_move)


    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    gen = logbook.select("gen")


    header = ["gen", "avg", "std", "min", "max"]

    import csv

    with open(r"./results/GP/" + str(i) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value],
                             ])


import pickle

print(move_count)
with open('./results/GP/max_moves.pkl', 'wb') as fp:
    pickle.dump(move_count, fp)

with open('./results/GP/best_phenotype_each_run.pkl', 'wb') as fp:
    pickle.dump(best_phenotype_each_run, fp)


