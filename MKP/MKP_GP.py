import operator
import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp
from deap import algorithms

# Load MKP instance (your same loader)
def MKPpopulate(filename):
    with open(filename, 'r') as f:
        x = f.read().split()
    NumColumns, NumRows, BestOF = int(x.pop(0)), int(x.pop(0)), float(x.pop(0))
    print(f'{NumColumns} items, {NumRows} knapsacks')
    c = np.array([float(x.pop(0)) for _ in range(NumColumns)])
    A = np.array([float(x.pop(0)) for _ in range(NumRows * NumColumns)]).reshape((NumRows, NumColumns))
    b = np.array([float(x.pop(0)) for _ in range(NumRows)])
    return c, A, b

# Parameters

profit_vector, knapsack_item_weight_matrix, knapsack_capacity_vector = MKPpopulate('mknap01_7.txt')
num_items = len(profit_vector)



def safe_add(left: str, right: str) -> str:
    return str(left) + "_" + str(right)

pset = gp.PrimitiveSetTyped("MAIN", [], str)  # No arguments; returns str

# Register the safe_add function that works on strings
pset.addPrimitive(safe_add, [str, str], str)

# Terminal generator: randomly choose an int in [1, 50] and convert to str
# def random_terminal():
#     return str(random.randint(0, 49))
#
# pset.addEphemeralConstant("randInt", random_terminal, str)
for i in range(50):
    pset.addTerminal(str(i), str)
#

# Convert trees to flat lists of item indices
def fitness_eval(individual, profit_vector, knapsack_capacity_vector, knapsack_item_weight_matrix):

    try:
        expr = toolbox.compile(expr=individual)
        # print(expr)

        items = [int(i) for i in expr.split("_")]
        item_list = list(set(items))
        # print(items)
    except:
        return 0,
    profit = 0
    knapsack_weight = [0, 0, 0, 0, 0]
    for item in item_list:
        profit = profit + profit_vector[item]
        item_weight = list(knapsack_item_weight_matrix[: , item])
        knapsack_weight = [a + b for a, b in zip(knapsack_weight, item_weight)]

    for i in range(len(knapsack_capacity_vector)):
        if knapsack_weight[i] > knapsack_capacity_vector[i]:
            return 0,

    return profit,

# DEAP Creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", fitness_eval, profit_vector=profit_vector,
                 knapsack_item_weight_matrix=knapsack_item_weight_matrix,
                 knapsack_capacity_vector=knapsack_capacity_vector)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# GP Run


hof = tools.HallOfFame(3)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("max", np.max)
stats.register("min", np.min)


N_RUNS = 30
collect_best = []
for i in range(N_RUNS):
    print()
    print()
    print("Run:", i)
    print()

    RANDOM_SEED = i
    np.random.seed(RANDOM_SEED)

    random.seed(RANDOM_SEED)
    pop = toolbox.population(n=200)

    pop, logbook = algorithms.eaSimple(pop, toolbox, 0.9, 0.01, 100, stats, halloffame=hof, verbose=True)

# Print best solution
    best_ind = hof[0]


    collect_best.append(hof.items[0].fitness.values[0])

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

with open(r"./results/GP/" + str(i) + "best.pickle", "wb") as f:
    pickle.dump(collect_best, file=f)


