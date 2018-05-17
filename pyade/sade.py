import numpy as np
import pyade.commons
from typing import Union, Callable, Dict, Any


def get_default_params(dim: int) -> dict:
    """
    Returns the default parameters of the Self-adaptive Differential Evolution Algorithm (SaDE).
    :param dim: Size of the problem (or individual).
    :type dim: int
    :return: Dict with the default parameters of SaDe
    :rtype dict
    """
    return {'max_evals': 10000 * dim, 'population_size': 10 * dim, 'callback': None,
            'individual_size': dim, 'seed': None}


def apply(population_size: int, individual_size: int,
          bounds: np.ndarray,
          func: Callable[[np.ndarray], float],
          callback: Callable[[Dict], Any],
          max_evals: int, seed: Union[int, None]):
    """
    Applies the Self-adaptive differential evolution algorithm (SaDE).
    :param population_size: Size of the population.
    :type population_size: int
    :param individual_size: Number of gens/features of an individual.
    :type individual_size: int
    :param bounds: Numpy ndarray with individual_size rows and 2 columns.
    First column represents the minimum value for the row feature.
    Second column represent the maximum value for the row feature.
    :type bounds: np.ndarray
    :param func: Evaluation function. The function used must receive one
     parameter.This parameter will be a numpy array representing an individual.
    :type func: Callable[[np.ndarray], float]
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param max_evals: Number of evaluatios after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]
    """

    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer.")

    if type(individual_size) is not int or individual_size <= 0:
        raise ValueError("individual_size must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    # 1. Initialization
    np.random.seed(seed)
    population = pyade.commons.init_population(population_size, individual_size, bounds)

    # 2. SaDE Algorithm
    probability = 0.5
    fitness = pyade.commons.apply_fitness(population, func)
    cr_m = 0.5
    f_m = 0.5

    sum_ns1 = 0
    sum_nf1 = 0
    sum_ns2 = 0
    sum_nf2 = 0
    cr_list = []

    f = np.random.normal(f_m, 0.3, population_size)
    f = np.clip(f, 0, 2)

    cr = np.random.normal(cr_m, 0.1, population_size)
    cr = np.clip(cr, 0, 1)

    max_iters = max_evals // population_size
    for current_generation in range(max_iters):
        # 2.1 Mutation
        # 2.1.1 Randomly choose which individuals do each mutation
        choice = np.random.rand(population_size)
        choice_1 = choice < probability
        choice_2 = choice >= probability

        # 2.1.2 Apply the mutations
        mutated = population.copy()
        mutated[choice_1] = pyade.commons.binary_mutation(population[choice_1],
                                                          f[choice_1].reshape(sum(choice_1), 1), bounds)
        mutated[choice_2] = pyade.commons.current_to_best_2_binary_mutation(population[choice_2],
                                                                            fitness[choice_2],
                                                                            f[choice_2].reshape(sum(choice_2), 1),
                                                                            bounds)

        # 2.2 Crossover
        crossed = pyade.commons.crossover(population, mutated, cr.reshape(population_size, 1))
        c_fitness = pyade.commons.apply_fitness(crossed, func)

        # 2.3 Selection
        population = pyade.commons.selection(population, crossed, fitness, c_fitness)
        winners = c_fitness < fitness
        fitness[winners] = c_fitness[winners]

        # 2.4 Self Adaption
        chosen_1 = np.sum(np.bitwise_and(choice_1, winners))
        chosen_2 = np.sum(np.bitwise_and(choice_2, winners))
        sum_ns1 += chosen_1
        sum_ns2 += chosen_2
        sum_nf1 += np.sum(choice_1) - chosen_1
        sum_nf2 += np.sum(choice_2) - chosen_2
        cr_list = np.concatenate((cr_list, cr[winners]))

        # 2.4.1 Adapt mutation strategy probability
        if (current_generation + 1) % 50 == 0:
            probability = sum_ns1 * (sum_ns2 + sum_nf2) / (sum_ns2 * (sum_ns1 + sum_nf1))
            probability = np.clip(probability, 0, 1)
            if np.isnan(probability):
                probability = .99
            sum_ns1 = 0
            sum_ns2 = 0
            sum_nf1 = 0
            sum_nf2 = 0

        # 2.4.2
        if (current_generation + 1) % 25 == 0:
            if len(cr_list) != 0:
                cr_m = np.mean(cr_list)
                cr_list = []
            cr = np.random.normal(cr_m, 0.1, population_size)
            cr = np.clip(cr, 0, 1)

        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best]
