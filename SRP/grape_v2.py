# -*- coding: utf-8 -*-


import re
import math
import copy
import numpy as np
import random
import copy



def median_abs_deviation(arr, axis=0):
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Calculate the median along axis 0
    median = np.median(arr, axis=0)

    # Calculate the absolute deviations from the median along axis 0
    abs_deviations = np.abs(arr - median)

    # Calculate the median of the absolute deviations along axis 0
    mad = np.median(abs_deviations, axis=0)

    return mad

class Individual(object):
    """
    A GE individual.
    """

    def __init__(self, genome, grammar, max_depth, codon_consumption, function_dict):
        """
        """
        
        self.genome = genome
        self.length = len(genome)
        if codon_consumption == 'lazy':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_lazy(genome, grammar, max_depth, function_dict)
        elif codon_consumption == 'eager':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_eager(genome, grammar, max_depth)
        else:
            raise ValueError("Unknown mapper")
            
        self.fitness_each_sample = []
        self.n_cases = 0


class Grammar(object):
    def __init__(self, file_address, sep="---", attribute=True):
        """Initializes the Grammar object by loading and parsing the grammar file."""
        if attribute:
            self.bnf_grammar, self.variable = self._load_and_clean_attribute_grammar(file_address, sep)
        self.bnf_grammar = self._load_and_clean_grammar(file_address)
        self.non_terminals = self._extract_non_terminals_and_change_bnf_grammar()
        self.start_rule = self.non_terminals[0]
        self.production_rules = self._create_production_rules()
        self.n_rules = [len(rules) for rules in self.production_rules]
        self._check_recursiveness_in_rules()
        self.PR_depth_to_terminate, self.part_PR_depth_to_terminate, self.max_depth_each_PR, self.min_depth_each_PR = self._calculate_depth_termination()
        self.initial_next_NT = re.search(r"\<(\w+)\>", self.start_rule).group()
        n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>", self.start_rule)])
        self.initial_list_depth = [1] * n_starting_NTs
        # If there is a codon consumption increase the minimum depth by the number of codon consumed
        # self._process_min_depth_of_functional_attribute()

    def _process_min_depth_of_functional_attribute(self):
        for i in range(len(self.production_rules)):
            for j in range(len(self.production_rules[i])):
                functional_attributes = self._find_functional_attributes(self.production_rules[i][j][0])
                if functional_attributes:
                    for functional_attribute in functional_attributes:
                        if len(functional_attribute['content'].strip().split(" ")) == 3:
                            self.production_rules[i][j][-1] = self.production_rules[i][j][-1] + int(functional_attribute['content'].strip().split(" ")[-1])
                            if self.min_depth_each_PR[i] > self.production_rules[i][j][-1]:
                                self.min_depth_each_PR[i] = self.production_rules[i][j][-1]
                            if self.max_depth_each_PR[i] < self.production_rules[i][j][-1]:
                                self.max_depth_each_PR[i] = self.production_rules[i][j][-1]

    def _find_functional_attributes(self, string):
        # Use regular expression to find the curly braces and their contents
        pattern = r'\{(.*?)\}'

        matches = []
        for match in re.finditer(pattern, string):
            start, end = match.span()  # Get the position of the curly braces
            content = match.group(1)  # Extract the content inside the curly braces
            matches.append({
                'start': start,
                'end': end,
                'content': content
            })

        return matches

    def _load_and_clean_attribute_grammar(self, file_address, sep):
        """Reads the grammar file, cleans up whitespace, and returns the cleaned grammar."""
        with open(file_address, "r") as text_file:
            bnf_grammar = text_file.read()
        split_grammar = bnf_grammar.split(sep)
        variable_dict = {}
        for var in split_grammar[0].strip().split("\n"):
            variable_dict[var.split("=")[0].strip()] = int(var.split("=")[1])

        return re.sub(r"\s+", " ", split_grammar[1]), variable_dict

    def _load_and_clean_grammar(self, file_address):
        """Reads the grammar file, cleans up whitespace, and returns the cleaned grammar."""
        with open(file_address, "r") as text_file:
            bnf_grammar = text_file.read()
        return re.sub(r"\s+", " ", bnf_grammar)  # Remove extra spaces

    def _extract_non_terminals_and_change_bnf_grammar(self):
        """Extracts all non-terminal symbols from the grammar."""
        non_terminals = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>\s*::=", self.bnf_grammar)]
        for nt in non_terminals:
            self.bnf_grammar = self.bnf_grammar.replace(nt + " ::=", " ::=")  # Remove NT definition part
        return non_terminals

    def _create_production_rules(self):
        """Splits the grammar into production rules and categorizes them as terminal or non-terminal."""
        rules = self.bnf_grammar.split("::=")[1:]
        rules = [rule.strip().replace('\n', '').replace('\t', '') for rule in rules]
        production_rules = [rule.split('|') for rule in rules]

        for i in range(len(production_rules)):
            production_rules[i] = [self._categorize_rule(item.strip(), j) for j, item in enumerate(production_rules[i])]
        return production_rules

    def _categorize_rule(self, rule, j):
        """Categorizes a single rule as either terminal or non-terminal."""
        if re.findall(r"\<([\(\)\w,-.]+)\>", rule):
            arity = len(re.findall(r"\<([\(\)\w,-.]+)\>", rule))  # Count NT occurrences
            return [rule, "non-terminal", arity, j]
        else:
            return [rule, "terminal", 0, j]  # Terminal rules have arity 0

    def _check_recursiveness_in_rules(self):
        """Checks if the production rules are recursive."""
        for i in range(len(self.production_rules)):
            for j in range(len(self.production_rules[i])):
                NTs_to_check = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[i][j][0])
                NTs_to_check = ['<' + item + '>' for item in NTs_to_check]
                unique_NTs = np.unique(NTs_to_check, return_counts=False)
                self.production_rules[i][j].append(
                    self._is_recursive(unique_NTs=unique_NTs, stack=[self.non_terminals[i]]))

    def _is_recursive(self, unique_NTs, stack):
        """Checks if a rule is recursive by traversing the production tree."""
        for NT in unique_NTs:
            if NT in stack:
                return True
            else:
                stack.append(NT)
                recursive = self._check_recursiveness(NT=NT, stack=stack)
                if recursive:
                    return True
                stack.pop()
        return False

    def _check_recursiveness(self, NT, stack):
        """Checks recursiveness for a specific non-terminal."""
        idx_NT = self.non_terminals.index(NT)
        for rule in self.production_rules[idx_NT]:
            NTs_to_check = re.findall(r"\<([\(\)\w,-.]+)\>", rule[0])
            NTs_to_check = ['<' + item_ + '>' for item_ in NTs_to_check]
            unique_NTs = np.unique(NTs_to_check, return_counts=False)
            if self._is_recursive(unique_NTs, stack):
                return True
        return False

    def _initialize_depth_tracking(self):
        """Initializes structures to track depth termination."""
        part_PR_depth_to_terminate = []
        isolated_non_terminal = []
        NT_depth_to_terminate = [None] * len(self.non_terminals)

        for i in range(len(self.production_rules)):
            part_PR_depth_to_terminate.append([])
            isolated_non_terminal.append([])

            for j in range(len(self.production_rules[i])):
                part_PR_depth_to_terminate[i].append([])
                isolated_non_terminal[i].append([])

                if self.production_rules[i][j][1] == 'terminal':
                    isolated_non_terminal[i][j].append(None)
                    part_PR_depth_to_terminate[i][j] = 1
                    if not NT_depth_to_terminate[i]:
                        NT_depth_to_terminate[i] = 1
                else:
                    for k in range(self.production_rules[i][j][2]):  # arity
                        part_PR_depth_to_terminate[i][j].append([])
                        term = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[i][j][0])[k]
                        isolated_non_terminal[i][j].append('<' + term + '>')
        return part_PR_depth_to_terminate, isolated_non_terminal, NT_depth_to_terminate

    def _calculate_depth_termination(self):
        """Calculates the minimum depth required to terminate the production rules."""
        part_PR_depth_to_terminate, isolated_non_terminal, NT_depth_to_terminate = self._initialize_depth_tracking()

        continue_ = True
        while continue_:
            if None not in NT_depth_to_terminate:
                continue_ = False

            for i in range(len(self.non_terminals)):
                for j in range(len(self.production_rules)):
                    for k in range(len(self.production_rules[j])):
                        for l in range(len(isolated_non_terminal[j][k])):
                            if self.non_terminals[i] == isolated_non_terminal[j][k][l]:
                                if NT_depth_to_terminate[i]:
                                    if not part_PR_depth_to_terminate[j][k][l]:
                                        part_PR_depth_to_terminate[j][k][l] = NT_depth_to_terminate[i] + 1
                                        if [] not in part_PR_depth_to_terminate[j][k]:
                                            if not NT_depth_to_terminate[j]:
                                                NT_depth_to_terminate[j] = part_PR_depth_to_terminate[j][k][l]

        PR_depth_to_terminate, max_depth_each_PR, min_depth_each_PR = self._finalize_depth_termination(
            part_PR_depth_to_terminate)
        return PR_depth_to_terminate, part_PR_depth_to_terminate, max_depth_each_PR, min_depth_each_PR

    def _finalize_depth_termination(self, part_PR_depth_to_terminate):
        """Finalizes the depth termination process and returns the max/min depth for each rule."""
        PR_depth_to_terminate = []
        max_depth_each_PR = []
        min_depth_each_PR = []

        for i in range(len(part_PR_depth_to_terminate)):
            for j in range(len(part_PR_depth_to_terminate[i])):
                depth_ = max(part_PR_depth_to_terminate[i][j]) if isinstance(part_PR_depth_to_terminate[i][j],
                                                                             list) else part_PR_depth_to_terminate[i][j]
                PR_depth_to_terminate.append(depth_)
                self.production_rules[i][j].append(depth_)

        for PR in self.production_rules:
            choices_ = [choice[5] for choice in PR]
            max_depth_each_PR.append(max(choices_))
            min_depth_each_PR.append(min(choices_))

        return PR_depth_to_terminate, max_depth_each_PR, min_depth_each_PR


def mapper(genome, grammar, max_depth):
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])    
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_eager(genome, grammar, max_depth):
    """
    Identical to the previous one.
    TODO Solve the names later.
    """    

    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure


def initialize_depth(phenotype):
    """Initializes the depth list based on the number of starting non-terminals."""
    n_starting_NTs = len(re.findall(r"\<([\(\)\w,-.]+)\>", phenotype))
    return [1] * n_starting_NTs


def choose_production_rule(NT_index, grammar, idx_genome, structure, genome):
    """Chooses a production rule based on the current non-terminal (NT) and the genome."""
    if grammar.n_rules[NT_index] == 1:
        return 0, idx_genome, structure  # Only one production rule for this NT, no codon consumed
    else:
        chosen_rule = genome[idx_genome] % grammar.n_rules[NT_index]
        structure.append(chosen_rule)
        idx_genome += 1  # Move to the next codon
        return chosen_rule, idx_genome, structure


def replace_non_terminal(NT, rule, phenotype):
    """Replaces the first occurrence of a non-terminal (NT) in the phenotype with the chosen production rule."""
    phenotype = phenotype.replace(NT, rule, 1)
    return phenotype


def check_arity_and_update_depth(NT_index, chosen_rule, grammar, idx_depth, nodes, list_depth):
    """Checks the arity of the chosen rule and updates the depth accordingly."""
    arity = grammar.production_rules[NT_index][chosen_rule][2]

    if arity == 0:  # Terminal (no non-terminals)
        idx_depth += 1
        nodes += 1
    if arity == 1:
        return True, idx_depth, nodes, list_depth
    elif arity > 1:  # Production rule with more than one non-terminal
        if idx_depth == 0:
            list_depth = [list_depth[idx_depth]] * arity + list_depth[idx_depth + 1:]
        else:
            list_depth = list_depth[:idx_depth] + [list_depth[idx_depth]] * arity + list_depth[idx_depth + 1:]
    return False, idx_depth, nodes, list_depth


def get_next_non_terminal(phenotype):
    """Finds the next non-terminal (NT) in the current phenotype string."""
    next_ = re.search(r"\<(\w+)\>", phenotype)
    return next_.group() if next_ else None


def find_functional_attributes(string):
    # Use regular expression to find the curly braces and their contents
    pattern = r'\{(.*?)\}'

    matches = []
    for match in re.finditer(pattern, string):
        start, end = match.span()  # Get the position of the curly braces
        content = match.group(1)  # Extract the content inside the curly braces
        matches.append({
            'start': start,
            'end': end,
            'content': content
        })

    return matches


def find_replacement(functional_attributes, variable_dict, function_dict, idx_genome, structure, genome, phenotype):
    """Replace functional attributes with actual values"""
    replacement = []
    # print("In find replacement function before: ", structure)
    for i in functional_attributes:
        if len(i['content'].split(" ")) == 2:
            # no condon to be used
            variable_dict[i['content'].split(" ")[1]] = function_dict[i['content'].split(" ")[0]](
                variable_dict[i['content'].split(" ")[1]])
            replacement.append(["", False])
        if i['content'].strip() in variable_dict.keys():
            # replacement.append( i['content'].strip() + "_" + str(variable_dict[i['content']]))
            replacement.append([i['content'].strip() + str(variable_dict[i['content']]), False])
        if len(i['content'].split(" ")) == 3:
            # codons to be used
            split_function = i['content'].split(" ")
            number_of_codons_required = split_function[2]  # need to work on this when more than one codon is required in the function
            codon = genome[idx_genome]
            rep = i['content'].split(" ")[1] + str(
                function_dict[split_function[0]](variable_dict[split_function[1]], codon)[0])
            replacement.append([rep, True])

            idx_genome += 1
            structure.append(function_dict[split_function[0]](variable_dict[split_function[1]], codon)[1])

        # Finally add the condition that a function can use phenotype for its
        # analysis of the current state of the phenotype and update the phenotype
        if len(i['content'].split(" ")) > 3:
            split_function = i['content'].split(" ")
            rep, no_of_codon_used, remainders, possible_choices = function_dict[split_function[0]](phenotype, idx_genome, genome)

            replacement.append([rep, True])
            idx_genome += no_of_codon_used
            structure.extend(remainders)
            # print("In find replacement function after: ", structure)

    return replacement, idx_genome, structure


def replace_substrings(string, replacements, new_contents):
    """Replace substrings in a given string"""
    # Convert string to a list for mutable string operations
    # print(new_contents)
    # print(replacements)
    result = list(string)

    for i, replacement in enumerate(replacements):
        replacement['content'] = new_contents[i][0]

    # Sort the replacement dictionaries in reverse order to prevent index shift issues
    replacements = sorted(replacements, key=lambda x: x['start'], reverse=True)

    # Iterate through replacements and apply them
    for replacement in replacements:
        start = replacement['start']
        end = replacement['end']
        # Get corresponding replacement content

        # Replace the portion between start and end
        result[start:end] = replacement['content']

    # Join the list back into a string
    return ''.join(result)




def substitute_attribute(phenotype, idx_genome, structure, genome, variable_dict, function_dict):
    functional_attributes = find_functional_attributes(phenotype)
    if functional_attributes:
        replacement, idx_genome, structure = find_replacement(functional_attributes, variable_dict, function_dict,
                                                              idx_genome, structure, genome, phenotype)
        # print(functional_attributes)
        # print("within substitute_attribute", structure)
        for i in replacement:
            # print("replacement: ", i)
            if i[1]:
                phenotype = replace_substrings(phenotype, functional_attributes, replacement)
                return phenotype, idx_genome, structure, 1

        phenotype = replace_substrings(phenotype, functional_attributes, replacement)
        return phenotype, idx_genome, structure, 0
    else:
        return phenotype, idx_genome, structure, 0




def mapper_lazy(genome, grammar, max_depth, function_dict):
    """ Maps the genome to a phenotype based on the given grammar and returns the result."""

    variable_dict = copy.deepcopy(grammar.variable)
    # deepcopy the grammar variable
    idx_genome = 0  # Tracks the current index of the genome being used
    phenotype = grammar.start_rule  # Start with the initial rule
    list_depth = initialize_depth(phenotype)  # Initialize depth list for the starting non-terminals
    idx_depth = 0  # Tracks current depth index
    nodes = 0  # Counts the number of nodes generated in the tree
    structure = []
    # Keeps track of the structure used during mapping (basically the remainders)

    next_NT = get_next_non_terminal(phenotype)

    # len(genome) - 1 because two codons can be consumed at a time
    while next_NT and idx_genome < len(genome)-1:
        NT_index = grammar.non_terminals.index(next_NT)
        chosen_rule, idx_genome, structure = choose_production_rule(NT_index, grammar, idx_genome, structure, genome)

        # substitute {} by using function dict


        # Replace the non-terminal in the phenotype with the chosen production rule

        phenotype = replace_non_terminal(next_NT, grammar.production_rules[NT_index][chosen_rule][0], phenotype)

        # phenotype, idx_genome, structure, increase_depth = substitute_attribute(phenotype, idx_genome, structure,
        #                                                                         genome, variable_dict,
        #                                                                         function_dict)

        try:
            phenotype, idx_genome, structure, increase_depth = substitute_attribute(phenotype, idx_genome, structure,
                                                                                    genome, variable_dict,
                                                                                    function_dict)
        except IndexError:
            break

        # Check arity and update depth
        list_depth[idx_depth] += 1

        if list_depth[idx_depth] > max_depth:  # Stop if we exceed the maximum allowed depth
            break
        arity_1, idx_depth, nodes, list_depth = check_arity_and_update_depth(NT_index, chosen_rule, grammar, idx_depth,
                                                                             nodes, list_depth)
        if arity_1:
            pass
        # Get the next non-terminal to process
        next_NT = get_next_non_terminal(phenotype)


    # phenotype, idx_genome, structure, increase_depth = substitute_attribute(phenotype, idx_genome, structure, genome,
    #                                                                         variable_dict,
    #                                                                         function_dict)
    invalid = bool(next_NT)
    try:
        phenotype, idx_genome, structure, increase_depth = substitute_attribute(phenotype, idx_genome, structure,
                                                                                genome, variable_dict,
                                                                                function_dict)
    except IndexError:
        invalid = True

    # invalid = bool(next_NT)
    used_codons = idx_genome if not invalid else 0
    depth = max(list_depth)

    # Return the mapping result
    # print("Within mapping", structure)
    return phenotype, nodes, depth, used_codons, invalid, 0, structure


def random_initialisation(ind_class, pop_size, bnf_grammar, min_init_genome_length, max_init_genome_length, max_init_depth, codon_size, codon_consumption, genome_representation):
        """
        
        """
        population = []
        
        for i in range(pop_size):
            genome = []
            init_genome_length = random.randint(min_init_genome_length, max_init_genome_length)
            for j in range(init_genome_length):
                genome.append(random.randint(0, codon_size))
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            population.append(ind)
    
        if genome_representation == 'list':
            return population
        else:
            raise ValueError("Unkonwn genome representation")


def find_replacement_in_sensible_init(functional_attributes, variable_dict, function_dict, codon_size, phenotype):
    replacement = []
    remainder, len_possible_choices = None, None
    for i in functional_attributes:
        if len(i['content'].split(" ")) == 2:
            # no condon to be used
            variable_dict[i['content'].split(" ")[1]] = function_dict[i['content'].split(" ")[0]](
                variable_dict[i['content'].split(" ")[1]])
            replacement.append(["", False])
        if i['content'].strip() in variable_dict.keys():

            replacement.append([i['content'].strip() + str(variable_dict[i['content']]), False])
        if len(i['content'].split(" ")) == 3:

            split_function = i['content'].split(" ")
            number_of_codons_required = split_function[2]  # need to work on this when more than one codon is required in the function
            len_possible_choices = variable_dict[split_function[1]]
            remainder = random.randint(0, len_possible_choices - 1)
            codon = (random.randint(0, 1e10) % math.floor(((codon_size + 1) / len_possible_choices)) *
                     len_possible_choices) + remainder
            rep = i['content'].split(" ")[1] + str(
                function_dict[split_function[0]](variable_dict[split_function[1]], codon)[0])

            replacement.append([rep, True])

        # Finally add the condition that a function can use phenotype for its
        # analysis of the current state of the phenotype and update the phenotype
        if len(i['content'].split(" ")) > 3:
            split_function = i['content'].split(" ")
            # In the function we have to define our codon by using no of choices and remainder as shown above
            rep, no_of_codon_used, remainder, len_possible_choices = function_dict[split_function[0]](phenotype, None, None)

            replacement.append([rep, True])
            # print("Here", remainder, len_possible_choices)

    return replacement, remainder, len_possible_choices


def substitute_attribute_in_sensible_init(phenotype, remainders, possible_choices, variable_dict, function_dict, codon_size):
    functional_attributes = find_functional_attributes(phenotype)
    if functional_attributes:
        replacement, remainder, possible_choice = find_replacement_in_sensible_init(functional_attributes,
                                                                                    variable_dict, function_dict, codon_size, phenotype)
        # print(functional_attributes, replacement)
        phenotype = replace_substrings(phenotype, functional_attributes, replacement)
        if remainder != None and type(remainder) == int:
            # print("remainder in sensible init and creating codon:", remainder )
            remainders.append(remainder)
            possible_choices.append(possible_choice)

        if remainder != None and type(remainder) == list:
            # print("remainder list in sensible init and creating codon:", remainder )
            remainders.extend(remainder)
            possible_choices.extend(possible_choice)

        return phenotype, remainders, possible_choices
    else:
        return phenotype, remainders, possible_choices


def sensible_initialisation(ind_class, pop_size, bnf_grammar, min_init_depth,
                            max_init_depth, codon_size, codon_consumption,
                            genome_representation, function_dict):
    """

    """
    # Calculate the number of individuals to be generated with each method

    is_odd = pop_size % 2
    n_grow = int(pop_size / 2)

    n_sets_grow = max_init_depth - min_init_depth + 1
    set_size = int(n_grow / n_sets_grow)
    remaining = n_grow % n_sets_grow

    n_full = n_grow + is_odd + remaining  # if pop_size is odd, generate an extra ind with "full"

    # TODO check if it is possible to generate inds with max_init_depth

    population = []
    # Generate inds using "Grow"
    for i in range(n_sets_grow):
        max_init_depth_ = min_init_depth + i
        for j in range(set_size):
            variable_dict = copy.deepcopy(bnf_grammar.variable)
            remainders = []  # it will register the choices
            possible_choices = []  # it will register the respective possible choices

            phenotype = bnf_grammar.start_rule
            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>", phenotype)]  #
            depths = [1] * len(remaining_NTs)  # it keeps the depth of each branch
            idx_branch = 0  # index of the current branch being grown
            while len(remaining_NTs) != 0:
                idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if
                                  PR[5] + depths[idx_branch] <= max_init_depth_]
                Ch = random.choice(actual_options)
                phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                depths[idx_branch] += 1
                if codon_consumption == 'eager':
                    remainders.append(Ch[3])
                    possible_choices.append(len(total_options))
                elif codon_consumption == 'lazy':
                    if len(total_options) > 1:
                        remainders.append(Ch[3])
                        possible_choices.append(len(total_options))

                phenotype, remainders, possible_choices = substitute_attribute_in_sensible_init(phenotype, remainders, possible_choices, variable_dict, function_dict, codon_size)

                # print(phenotype, remainders, possible_choices)

                if Ch[2] > 1:
                    if idx_branch == 0:
                        depths = [depths[idx_branch], ] * Ch[2] + depths[idx_branch + 1:]
                    else:
                        depths = depths[0:idx_branch] + [depths[idx_branch], ] * Ch[2] + depths[idx_branch + 1:]
                if Ch[1] == 'terminal':
                    idx_branch += 1

                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>", phenotype)]

            phenotype, remainders, possible_choices = substitute_attribute_in_sensible_init(phenotype, remainders,
                                                                                            possible_choices,
                                                                                            variable_dict,
                                                                                            function_dict, codon_size)

            # Generate the genome
            genome = []

            if codon_consumption == 'eager' or codon_consumption == 'lazy':
                for k in range(len(remainders)):
                    codon = (random.randint(0, 1e10) % math.floor(((codon_size + 1) / possible_choices[k])) *
                             possible_choices[k]) + remainders[k]
                    genome.append(codon)
            else:
                raise ValueError("Unknown mapper")
            # print(genome)

            # Include a tail with 50% of the genome's size
            size_tail = max(int(0.5 * len(genome)),
                            1)  # Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
            for j in range(size_tail):
                genome.append(random.randint(0, codon_size))

            # Initialise the individual and include in the population
            ind = ind_class(genome, bnf_grammar, max_init_depth_, codon_consumption, function_dict)

            # Check if the individual was mapped correctly
            # print("Uncomment line number 894")

            if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                # print("Remainder", remainders, ind.structure, "match",remainders != ind.structure)
                # print(ind.structure)
                # print(phenotype)
                # print(ind.phenotype)
                # print(genome)
                # print(max(depths), ind.depth)
                #
                # print(phenotype.replace("\\n", "\n"), "\n\n", ind.phenotype.replace("\\n", "\n"))
                raise Exception('error in the mapping')

            population.append(ind)

    for i in range(n_full):
        variable_dict = copy.deepcopy(bnf_grammar.variable)
        remainders = []  # it will register the choices
        possible_choices = []  # it will register the respective possible choices

        phenotype = bnf_grammar.start_rule
        remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>", phenotype)]  #
        depths = [1] * len(remaining_NTs)  # it keeps the depth of each branch
        idx_branch = 0  # index of the current branch being grown

        while len(remaining_NTs) != 0:
            idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
            total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
            actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if
                              PR[5] + depths[idx_branch] <= max_init_depth]
            recursive_options = [PR for PR in actual_options if PR[4]]
            if len(recursive_options) > 0:
                Ch = random.choice(recursive_options)
            else:
                Ch = random.choice(actual_options)
            phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
            depths[idx_branch] += 1
            if codon_consumption == 'eager':
                remainders.append(Ch[3])
                possible_choices.append(len(total_options))
            elif codon_consumption == 'lazy':
                if len(total_options) > 1:
                    remainders.append(Ch[3])
                    possible_choices.append(len(total_options))

            phenotype, remainders, possible_choices = substitute_attribute_in_sensible_init(phenotype, remainders,
                                                                                            possible_choices,
                                                                                            variable_dict,
                                                                                            function_dict, codon_size)

            if Ch[2] > 1:
                if idx_branch == 0:
                    depths = [depths[idx_branch], ] * Ch[2] + depths[idx_branch + 1:]
                else:
                    depths = depths[0:idx_branch] + [depths[idx_branch], ] * Ch[2] + depths[idx_branch + 1:]
            if Ch[1] == 'terminal':
                idx_branch += 1

            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>", phenotype)]

        phenotype, remainders, possible_choices = substitute_attribute_in_sensible_init(phenotype, remainders,
                                                                                        possible_choices, variable_dict,
                                                                                        function_dict, codon_size)

        # Generate the genome
        genome = []
        if codon_consumption == 'eager' or codon_consumption == 'lazy':
            for j in range(len(remainders)):
                codon = (random.randint(0, 1e10) % math.floor(((codon_size + 1) / possible_choices[j])) *
                         possible_choices[j]) + remainders[j]
                genome.append(codon)
        else:
            raise ValueError("Unknown mapper")

        # Include a tail with 50% of the genome's size
        if codon_consumption == 'eager' or codon_consumption == 'lazy':
            size_tail = max(int(0.5 * len(genome)),
                            1)
            # Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.

        for j in range(size_tail):
            genome.append(random.randint(0, codon_size))

        # Initialise the individual and include in the population
        ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption, function_dict)

        # Check if the individual was mapped correctly
        # # print("Uncomment line number 977")
        # print(remainders, ind.structure)
        # print(phenotype, ind.phenotype)
        if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
            raise Exception('error in the mapping')

        population.append(ind)

    if genome_representation == 'list':
        return population
    elif genome_representation == 'numpy':
        for ind in population:
            ind.genome = np.array(ind.genome)
        return population
    else:
        raise ValueError("Unkonwn genome representation")


def crossover_onepoint(parent0, parent1, bnf_grammar, max_depth, codon_consumption, 
                       invalidate_max_depth, function_dict,
                       genome_representation='list', max_genome_length=None ):
    """
    
    """
    if max_genome_length:
        raise ValueError("max_genome_length not implemented in this onepoint")
    
    if parent0.invalid: #used_codons = 0
        possible_crossover_codons0 = len(parent0.genome)
    else:
        possible_crossover_codons0 = min(len(parent0.genome), parent0.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid:
        possible_crossover_codons1 = len(parent1.genome)
    else:
        possible_crossover_codons1 = min(len(parent1.genome), parent1.used_codons)
#        print()
    
    parent0_genome = parent0.genome.copy()
    parent1_genome = parent1.genome.copy()
    continue_ = True
#    a = 0
    while continue_:
        #Set points for crossover within the effective part of the genomes
        point0 = random.randint(1, possible_crossover_codons0)
        point1 = random.randint(1, possible_crossover_codons1)
        
        if genome_representation == 'list':
            #Operate crossover
            new_genome0 = parent0_genome[0:point0] + parent1_genome[point1:]
            new_genome1 = parent1_genome[0:point1] + parent0_genome[point0:]
        else:
            raise ValueError("Only 'list' representation is implemented")
        
        new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption, function_dict)
        new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption, function_dict)
        
        if invalidate_max_depth: # In the mapping, if a ind surpasses max depth, it is invalid, and we won't redo crossover
            continue_ = False
        else: # We check if a ind surpasses max depth, and if so we will redo crossover
            continue_ = new_ind0.depth > max_depth or new_ind1.depth > max_depth


                      
        
    del new_ind0.fitness.values, new_ind1.fitness.values
    return new_ind0, new_ind1   
    
def mutation_int_flip_per_codon(ind, mut_probability, codon_size, bnf_grammar, max_depth, 
                                codon_consumption, invalidate_max_depth,
                                max_genome_length, function_dict):
    """

    """
    # Operation mutation within the effective part of the genome
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
    else:
        possible_mutation_codons = min(len(ind.genome), ind.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    continue_ = True
    
    genome = copy.deepcopy(ind.genome)
    mutated_ = False

    while continue_:
        for i in range(possible_mutation_codons):
            if random.random() < mut_probability:
                genome[i] = random.randint(0, codon_size)
                mutated_ = True
               # break
    
        new_ind = reMap(ind, genome, bnf_grammar, max_depth, codon_consumption, function_dict)
        
        if invalidate_max_depth: # In the mapping, if ind surpasses max depth, it is invalid, and we won't redo mutation
            continue_ = False
        else: # We check if ind surpasses max depth, and if so we will redo mutation
            continue_ = new_ind.depth > max_depth
        
#    if max_genome_length:
#        if new_ind.depth > max_depth or len(new_ind.genome) > max_genome_length:
#            return ind,
#        else:
#            return new_ind,
#    else:
        #if new_ind.depth > max_depth:
        #    return ind,
        #else:
    if mutated_:
        del new_ind.fitness.values
    return new_ind,
        
def reMap(ind, genome, bnf_grammar, max_tree_depth, codon_consumption, function_dict):
    #TODO refazer todo o reMap para nao copiar o ind
    #
    #ind = Individual(genome, bnf_grammar, max_tree_depth, codon_consumption)
    #ind = Individual(genome, bnf_grammar, max_tree_depth, codon_consumption)
    ind.genome = genome
    if codon_consumption == 'lazy':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_lazy(genome, bnf_grammar, max_tree_depth, function_dict)
    elif codon_consumption == 'eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_eager(genome, bnf_grammar, max_tree_depth)
    else:
        raise ValueError("Unknown mapper")
        
    return ind

def replace_nth(string, substring, new_substring, nth):
    find = string.find(substring)
    i = find != -1
    while find != -1 and i != nth:
        find = string.find(substring, find + 1)
        i += 1
    if i == nth:
        return string[:find] + new_substring + string[find+len(substring):]
    return string