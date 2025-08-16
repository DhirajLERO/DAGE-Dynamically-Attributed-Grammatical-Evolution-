import warnings
warnings.filterwarnings("ignore")

import grape
import algorithms

import copy
import random

from deap import creator, base, tools

import random
import matplotlib.pyplot as plt

problem = 'MultipleKnapsack_problem'

import numpy as np

GRAMMAR_FILE = "BNF_2.bnf"
BNF_GRAMMAR = grape.Grammar(r"grammar/" + GRAMMAR_FILE)


def MKPpopulate(name: str) -> tuple:
    '''
    returns:
        c -- objective function coefficients array (shape = 1 * n)
        A -- constraints coefficients matrix A (shape = m * n)
        b -- right hand side values (shape = 1 * m)
    '''

    # Opening .txt file to read raw data of an instance
    file = open(str(name), 'r')
    x = []
    for line in file:
        splitLine = line.split()
        for i in range(len(splitLine)):
            x.append(splitLine[i])
    file.close()

    # Define parameters
    NumColumns, NumRows, BestOF = int(x.pop(0)), int(x.pop(0)), float(x.pop(0))
    print('This instance has %d items and %d knapsacks' % (NumColumns, NumRows))

    if BestOF != float(0):
        print('Best known integer objective value for this instance = ', BestOF)
    else:
        print('Best integer objective value for this instance is not indicated')

    # Populating Objective Function Coefficients
    c = np.array([float(x.pop(0)) for i in range(NumColumns)])

    assert type(c) == np.ndarray
    assert len(c) == NumColumns

    # Populating A matrix (size NumRows * NumColumns)
    ConstCoef = np.array([float(x.pop(0)) for i in range(int(NumRows * NumColumns))])

    assert type(ConstCoef) == np.ndarray
    assert len(ConstCoef) == int(NumRows * NumColumns)

    A = np.reshape(ConstCoef, (NumRows, NumColumns))  # reshaping the 1-d ConstCoef into A

    assert A.shape == (NumRows, NumColumns)

    # Populating the RHS
    b = np.array([float(x.pop(0)) for i in range(int(NumRows))])

    assert len(b) == NumRows
    assert type(b) == np.ndarray

    return (c, A, b)


def fitness_eval(individual, points, profit_vector, knapsack_capacity_vector, knapsack_item_weight_matrix):

    if individual.invalid == True:
        return 0,
    else:
        item_list = list(set(int(i) for i in individual.phenotype.split("_")))
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


profit_vector, knapsack_item_weight_matrix, knapsack_capacity_vector = MKPpopulate('mknap01_7.txt')

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
# toolbox.register("populationCreator", grape.random_initialisation, creator.Individual)
# toolbox.register("populationCreator", grape.PI_Grow, creator.Individual)


toolbox.register("evaluate", fitness_eval, profit_vector=profit_vector,
                 knapsack_item_weight_matrix=knapsack_item_weight_matrix,
                 knapsack_capacity_vector=knapsack_capacity_vector)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=10)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 200
MAX_GENERATIONS = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.01
ELITE_SIZE = 3  # round(0.01*POPULATION_SIZE) #it should be smaller or equal to HALLOFFAME_SIZE
HALLOFFAME_SIZE = 3  # round(0.01*POPULATION_SIZE) #it should be at least 1

MIN_INIT_GENOME_LENGTH = 30  # used only for random initialisation
MAX_INIT_GENOME_LENGTH = 50
random_initilisation = False  # put True if you use random initialisation

MAX_INIT_TREE_DEPTH = 20 # equivalent to 6 in GP with this grammar
MIN_INIT_TREE_DEPTH = 7

MAX_TREE_DEPTH = 35  # equivalent to 17 in GP with this grammar
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
                'selection_time', 'generation_time', "out_of_weight_indi"]

N_RUNS = 30
collect_best = []
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
    print("Best individual: \n", "\n".join(textwrap.wrap(best, 80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    collect_best.append(hof.items[0].fitness.values[0])

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
    out_of_weight_indi = logbook.select("out_of_weight_indi")

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
                             generation_time[value], out_of_weight_indi[value]])

print(collect_best)