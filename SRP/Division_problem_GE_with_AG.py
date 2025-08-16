import warnings
warnings.filterwarnings("ignore")

import grape_v2 as grape
import algorithms_AG as algorithms


from os import path
import pandas as pd
import numpy as np

np.seterr(divide = 'raise')

from deap import creator, base, tools

import random
import matplotlib.pyplot as plt
import copy

problem = 'division_problem'

import numpy as np
import time


# def compute_y(x):
#     x = x.flatten()
#     term1 = x ** 3
#     # term2 = x ** 2
#     # term3 = x ** 1
#     term2 = np.exp(-x)
#     term3 = np.cos(x)
#     term4 = np.sin(x)
#     term5 = (np.sin(x) ** 2 * np.cos(x) - 1)
#
#     y = term1 * term2 * term3 * term4 * term5
#     return y


def compute_y(x):
    return np.arcsinh(x).flatten()



def generate_array(min_value, steps, total_number):

    array = np.linspace(min_value, min_value + steps * (total_number - 1), total_number)
    return array.reshape(1, -1)  # Reshape to 2D array with shape (1, total_number)

    # return np.linspace(min_value, min_value + steps * (total_number - 1), total_number)


def setDataSet(x_train_min, x_train_steps, x_train_count, x_test_min, x_test_steps, x_test_count, GRAMMAR_FILE, reg_fun):

    X_train = generate_array(x_train_min, x_train_steps, x_train_count)
    X_test = generate_array(x_test_min, x_test_steps, x_test_count)

    Y_train = reg_fun(X_train)
    Y_test = reg_fun(X_test)


    BNF_GRAMMAR = grape.Grammar(r"grammar/" + GRAMMAR_FILE)
    # BNF_GRAMMAR= None

    return X_train, Y_train, X_test, Y_test, BNF_GRAMMAR





def fitness_eval(individual, points):
    # points = [X, Y]
    x = points[0]
    y = points[1]

    # print(x[0])
    # exit()

    if individual.invalid == True:
        return np.NaN,

    try:
        # print(individual.phenotype)
        pred = eval(individual.phenotype)

    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        return np.nan,
    except Exception as err:
        # Other errors should not usually happen (unless we have
        # an unprotected operator) so user would prefer to see them.
        print("evaluation error", err)
        raise
    assert np.isrealobj(pred)

    try:
        fitness = np.mean(np.square(y - pred))
        # print(fitness)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError, ValueError):
        fitness = "inf"
    except Exception as err:
        # Other errors should not usually happen (unless we have
        # an unprotected operator) so user would prefer to see them.
        print("fitness error", err)
        raise

    if fitness == float("inf"):
        return np.nan,

    return fitness,


import sympy as sp


def check_zero_denominator(expression):
    try:
        # Define x as an indexed variable
        x = sp.IndexedBase('x')

        # Parse the expression
        expr = sp.sympify(expression, locals={'x': x})
        # print(expr)

        # Identify denominators in the expression
        for term in sp.preorder_traversal(expr):
            if isinstance(term, sp.Mul):  # Check for division
                for arg in term.args:
                    if arg.is_Pow and arg.exp == -1:  # Denominator is a power of -1
                        denominator = arg.base
                        # Substitute x[0] = 0 in the denominator
                        result = denominator.subs(x[0], 0)
                        if result == 0:
                            return True  # Denominator becomes zero
        return False  # No denominator becomes zero
    except Exception as e:
        return f"Error parsing the expression: {e}"

# def finalize_division_int(phenotype, idx_genome, genome):
#     idx_genome_temp = copy.deepcopy(idx_genome)
#     phenotype_temp = copy.deepcopy(phenotype)
#     phenotype_temp = phenotype_temp.replace("<e>", "0")
#     phenotype_temp = phenotype_temp.replace("<op>", "+")
#     phenotype_temp = phenotype_temp.replace("{finalize_division_int x[0] 1 phenotype}", "x[0]")
#
#     if idx_genome is not None:
#         if check_zero_denominator(phenotype_temp):
#             return "x[0] + 0.1", 0, [], []
#         else:
#             return "x[0]", 0, [], []
#     else:
#         if check_zero_denominator(phenotype_temp):
#             return "x[0] + 0.1", 0, [], []
#         else:
#             return "x[0]", 0, [], []

def finalize_division_int(phenotype, idx_genome, genome):
    idx_genome_temp = copy.deepcopy(idx_genome)
    phenotype_temp = copy.deepcopy(phenotype)
    phenotype_temp = phenotype_temp.replace("<e>", "0")
    phenotype_temp = phenotype_temp.replace("<op>", "+")
    phenotype_temp = phenotype_temp.replace("{finalize_division_int x[0] 1 phenotype}", "x[0]")

    if idx_genome is not None:
        if check_zero_denominator(phenotype_temp):
            return "x[0] + 1", 0, [], []
        else:
            return "x[0]", 0, [], []
    else:
        if check_zero_denominator(phenotype_temp):
            return "x[0] + 1", 0, [], []
        else:
            return "x[0]", 0, [], []

# def finalize_division_int(phenotype, idx_genome, genome):
#     idx_genome_temp = copy.deepcopy(idx_genome)
#     phenotype_temp = copy.deepcopy(phenotype)
#     phenotype_temp = phenotype_temp.replace("<e>", "0")
#     phenotype_temp = phenotype_temp.replace("<op>", "+")
#     phenotype_temp = phenotype_temp.replace("{finalize_division_int x[0] 1 phenotype}", "x[0]")
#
#     if idx_genome is not None:
#         if check_zero_denominator(phenotype_temp):
#             return "x[0] + 0.1", 1, [0], [1]
#         else:
#             return "x[0]", 1, [0], []
#     else:
#         if check_zero_denominator(phenotype_temp):
#             return "x[0] + 0.1", 1, [0], [1]
#         else:
#             return "x[0]", 1, [0], [1]

def find_const_no_div(l, genome, idx_genome):
    x = genome[idx_genome] % len(l)
    # print(x, len(l), genome[idx_genome])
    replacement = l[x]
    idx_genome = idx_genome +  1
    return str(replacement), idx_genome, [x], [len(l)]

def finalize_division_const(phenotype, idx_genome, genome):
    idx_genome_temp = copy.deepcopy(idx_genome)
    phenotype_temp = copy.deepcopy(phenotype)
    no_of_codon_used = 0
    phenotype_temp = phenotype_temp.replace("<e>", "0")
    phenotype_temp = phenotype_temp.replace("<op>", "+")
    phenotype_temp = phenotype_temp.replace("{finalize_division_const const 1 phenotype}", "0")

    if idx_genome is not None:
        if check_zero_denominator(phenotype_temp):
            l = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            replacement, idx_genome_temp, remainders, possible_choices = find_const_no_div(l, genome,
                                                                                           idx_genome_temp)
            # print("inside mapper initiation", remainders, possible_choices, genome[idx_genome_temp-1])
            return replacement, idx_genome_temp - idx_genome, remainders, possible_choices
        else:
            l = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            replacement, idx_genome_temp, remainders, possible_choices = find_const_no_div(l, genome,
                                                                                           idx_genome_temp)
            return replacement, idx_genome_temp - idx_genome, remainders, possible_choices

    else:
        if check_zero_denominator(phenotype_temp):
            l = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            remainder = random.randint(0, len(l) - 1)
            # print("inside sensible initiation", [remainder], [len(l)])
            return str(l[remainder]), no_of_codon_used, [remainder], [len(l)]
        else:
            l = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            remainder = random.randint(0, len(l) - 1)
            # print("inside sensible initiation", [remainder], [len(l)])
            return str(l[remainder]), no_of_codon_used, [remainder], [len(l)]



function_dict = {"finalize_division_int": finalize_division_int,
                 "finalize_division_const": finalize_division_const}



toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
# toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
# toolbox.register("populationCreator", grape.PI_Grow, creator.Individual)

toolbox.register("evaluate", fitness_eval)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=10)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 100
MAX_GENERATIONS = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.01
ELITE_SIZE = 3  # round(0.01*POPULATION_SIZE) #it should be smaller or equal to HALLOFFAME_SIZE
HALLOFFAME_SIZE = 3  # round(0.01*POPULATION_SIZE) #it should be at least 1


MIN_INIT_GENOME_LENGTH = 30  # used only for random initialisation
MAX_INIT_GENOME_LENGTH = 50
random_initilisation = False  # put True if you use random initialisation

MAX_INIT_TREE_DEPTH = 10 # equivalent to 6 in GP with this grammar
MIN_INIT_TREE_DEPTH = 6

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
                'selection_time', 'generation_time', "division_count"]

N_RUNS = 30
test_fitness = []
timing_for_each_run = []
final_solution = []

for i in range(N_RUNS):
    start_time = time.time()
    print()
    print()
    print("Run:", i)
    print()

    RANDOM_SEED = i

    np.random.seed(RANDOM_SEED)

    X_train, Y_train, X_test, Y_test, BNF_GRAMMAR = setDataSet(0, 1.0, 50, \
                                                               0.1, 0.25, 200, \
                                                               'AG.bnf', reg_fun=compute_y)

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, BNF_GRAMMAR)  # We set up this inside the loop for the case in which the data is defined randomly

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
                                               min_init_depth=MAX_INIT_TREE_DEPTH,
                                               max_init_depth=MIN_INIT_TREE_DEPTH,
                                               codon_size=CODON_SIZE,
                                               codon_consumption=CODON_CONSUMPTION,
                                               genome_representation=GENOME_REPRESENTATION,
                                               function_dict=function_dict
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
                                                            points_train=[X_train, Y_train],
                                                            points_test=[X_test, Y_test],
                                                            codon_consumption=CODON_CONSUMPTION,
                                                            report_items=REPORT_ITEMS,
                                                            genome_representation=GENOME_REPRESENTATION,
                                                            invalidate_max_depth=False,
                                                            stats=stats, halloffame=hof, verbose=False, function_dict=function_dict)
    timing_for_each_run.append(start_time - time.time())
    import textwrap

    best = hof.items[0].phenotype
    print("Best individual: \n", "\n".join(textwrap.wrap(best, 80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    print("Test Fitness: ", fitness_eval(hof.items[0], [X_test, Y_test])[0])
    test_fitness.append(fitness_eval(hof.items[0], [X_test, Y_test])[0])
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

    division_count = logbook.select("division_count")

    import csv

    r = RANDOM_SEED

    header = REPORT_ITEMS

    with open(r"./results/DAGEV1_div/" + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
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
                             generation_time[value], division_count[value]])

import pickle

with open('./results/DAGEV1_div/testfitness.pkl', 'wb') as fp:
    pickle.dump(test_fitness, fp)

with open("./results/DAGEV1_div/time.pkl", 'wb') as fp:
    pickle.dump(timing_for_each_run, fp)