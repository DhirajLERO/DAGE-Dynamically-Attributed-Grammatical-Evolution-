import warnings
warnings.filterwarnings("ignore")

import grape_v2 as grape
import algorithms_AG as algorithms
import re

from deap import creator, base, tools

import random
import matplotlib.pyplot as plt
import copy

problem = 'MultipleKnapsack_problem'

import numpy as np







GRAMMAR_FILE = "AG_2.bnf"
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


def search_used_items(text):

    matches = re.findall(r'\d+', text)
    # Convert each found match to an integer
    return list(set([int(num) for num in matches]))

def compute_needed_items(used_item, items, knapsack_item_weight_matrix, knapsack_capacity_vector):
    current_knapsack_weight = [0, 0, 0, 0, 0]
    if len(used_item) != 0:
        for item in used_item:
            item_weight = list(knapsack_item_weight_matrix[:, item])
            current_knapsack_weight = [a + b for a, b in zip(current_knapsack_weight, item_weight)]

    items_to_remove = []

    for item in items:
        item_weight = list(knapsack_item_weight_matrix[:, item])
        temp_knapsack_weight = [a + b for a, b in zip(current_knapsack_weight, item_weight)]
        for i in range(len(temp_knapsack_weight)):
            if temp_knapsack_weight[i] > knapsack_capacity_vector[i]:
                items_to_remove.append(item)

    needed_item = [item for item in items if item not in items_to_remove]
    return needed_item

def find_choice(directions, genome, idx_genome):
    x = genome[idx_genome] % len(directions)
    # print(x, len(l), genome[idx_genome])
    replacement = directions[x]
    idx_genome = idx_genome + 1
    return str(replacement), idx_genome, [x], [len(directions)]

# try choosing from already choosen once the weight limit is breached!
def finalize_item(phenotype, idx_genome, genome):

    global knapsack_item_weight_matrix
    global knapsack_capacity_vector
    # print(phenotype)
    items = list(range(50))
    idx_genome_temp = copy.deepcopy(idx_genome)
    phenotype_temp = copy.deepcopy(phenotype)
    used_item = search_used_items(phenotype_temp)
    no_of_codon_used = 0
    # print(items)

    if len(used_item) != 0:
        for i in used_item:
            items.remove(i)

    required_items = compute_needed_items(used_item, items,
                                          knapsack_item_weight_matrix,
                                          knapsack_capacity_vector)
    # print(used_item)
    # print(required_items)
    if idx_genome is not None:

        if len(required_items) == 0:
            replacement, idx_genome_temp, remainders, possible_choices = find_choice(items,
                                                                                     genome, idx_genome_temp)
            return replacement, idx_genome_temp - idx_genome, remainders, possible_choices
        else:
            replacement, idx_genome_temp, remainders, possible_choices = find_choice(required_items,
                                                                                     genome, idx_genome_temp)
            return replacement, idx_genome_temp - idx_genome, remainders, possible_choices
    else:
        if len(required_items) == 0:
            remainder = random.randint(0, len(items) - 1)
            # print("inside sensible initiation", [remainder], [len(l)])
            return str(items[remainder]), no_of_codon_used, [remainder], [len(items)]
        else:
            remainder = random.randint(0, len(required_items) - 1)
            # print("inside sensible initiation", [remainder], [len(l)])
            return str(required_items[remainder]), no_of_codon_used, [remainder], [len(required_items)]







function_dict = {"finalize_item": finalize_item}

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
  # We set up this inside the loop for the case in which the data is defined randomly

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
                                                            points_train=None,
                                                            points_test=None,
                                                            codon_consumption=CODON_CONSUMPTION,
                                                            report_items=REPORT_ITEMS,
                                                            genome_representation=GENOME_REPRESENTATION,
                                                            invalidate_max_depth=False,
                                                            stats=stats, halloffame=hof, verbose=False, function_dict=function_dict)

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

    out_of_weight_indi= logbook.select("out_of_weight_indi")

    import csv

    r = RANDOM_SEED

    header = REPORT_ITEMS

    with open(r"./results/DAGE/" + str(r) + "_AG.csv", "w", encoding='UTF8', newline='') as csvfile:
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
                             generation_time[value],out_of_weight_indi[value]])

print(collect_best)


# [15400.0, 15747.0, 15737.0, 15687.0, 15858.0, 16109.0, 15736.0, 15834.0, 15742.0, 15725.0]
# [15938.0, 15411.0, 15673.0, 15517.0, 15809.0, 15769.0, 15680.0, 15702.0, 15747.0, 15905.0]

# mutation in 0.01

# [16346.0, 16342.0, 16303.0, 16357.0, 16254.0, 16033.0, 16334.0, 16230.0, 16213.0, 16352.0, 16012.0, 16194.0, 16380.0, 16469.0, 16347.0, 16232.0, 16246.0, 16292.0, 16101.0, 16321.0, 15977.0, 16309.0, 16412.0, 16296.0, 15781.0, 16263.0, 16328.0, 16308.0, 16224.0, 16236.0]
# [16377.0, 16285.0, 16052.0, 16208.0, 15955.0, 16324.0, 16215.0, 15903.0, 15936.0, 15852.0, 16111.0, 15960.0, 16197.0, 15953.0, 16202.0, 15749.0, 15827.0, 16342.0, 15995.0, 16073.0, 15749.0, 16039.0, 16079.0, 16102.0, 16036.0, 16271.0, 16084.0, 16063.0, 15807.0, 15968.0]



