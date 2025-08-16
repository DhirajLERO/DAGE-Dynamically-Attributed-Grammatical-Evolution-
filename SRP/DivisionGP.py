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



def compute_y(x):
    return np.arcsinh(x).flatten()



def generate_array(min_value, steps, total_number):

    array = np.linspace(min_value, min_value + steps * (total_number - 1), total_number)
    return array.reshape(1, -1)  # Reshape to 2D array with shape (1, total_number)

    # return np.linspace(min_value, min_value + steps * (total_number - 1), total_number)


def setDataSet(x_train_min, x_train_steps, x_train_count, x_test_min, x_test_steps, x_test_count, reg_fun):

    X_train = generate_array(x_train_min, x_train_steps, x_train_count)
    X_test = generate_array(x_test_min, x_test_steps, x_test_count)

    Y_train = reg_fun(X_train)
    Y_test = reg_fun(X_test)
    return X_train, Y_train, X_test, Y_test

# def div(a, b):
#     if b!=0:
#         return a/b
#     else:
#         return 0

def div(a, b):
    return a / b

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

pset.addPrimitive(div, 2)

pset.addEphemeralConstant("rand5step", lambda: round(random.uniform(-5, 5) * 2) / 2)

pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)



import math

def fitness_eval(individual, points):
    # Transform the tree expression in a callable function
    x_all = points[0][0]
    # print(x)
    # exit()
    try:
        func = toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        sqerrors = ((func(x) - compute_y(x))**2 for x in x_all)
        return math.fsum(sqerrors) / len(x_all),
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return np.nan,
    except Exception as err:
        # Other errors should not usually happen (unless we have
        # an unprotected operator) so user would prefer to see them.
        print("fitness error", err)
        raise


X_train, Y_train, X_test, Y_test = setDataSet(0, 1.0, 50, \
                                                               0.1, 0.25, 200,  reg_fun=compute_y)


# DEAP Creator
toolbox.register("evaluate", fitness_eval, points=(X_train, Y_train))
toolbox.register("select", tools.selTournament, tournsize=10)


toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))



# GP Run


hof = tools.HallOfFame(3)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("max", np.max)
stats.register("min", np.min)


N_RUNS = 30
# collect_best = []

test_fitness = []
for i in range(N_RUNS):
    print()
    print()
    print("Run:", i)
    print()

    RANDOM_SEED = i
    np.random.seed(RANDOM_SEED)


    random.seed(RANDOM_SEED)
    pop = toolbox.population(n=100)

    pop, logbook = algorithms.eaSimple(pop, toolbox, 0.9, 0.01, 100, stats, halloffame=hof, verbose=True)

# Print best solution
    best_ind = hof[0]

    test = fitness_eval(best_ind, [X_test, Y_test])[0]

    fitness_test = [float('nan')] * 100
    fitness_test.append(test)


    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    gen = logbook.select("gen")
    # print(len(gen))
    # print(gen)
    # exit()


    header =  ["gen", "avg", "std", "min", "max", "fitness_test"]

    import csv

    with open(r"./results/GP_new/" + str(i) + "test.csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value], fitness_test[value]
                             ])


import pickle


with open("./results/GP/testfitness_test.pkl", 'wb') as fp:
    pickle.dump(test_fitness, fp)



