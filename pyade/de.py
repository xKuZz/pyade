import numpy as np
import pyade.commons
from typing import Callable, Union, Dict, Any


def get_default_de_params() -> dict:
    """
    Returns the default parameters of the Differential Evolution Algorithm
    :return: Dict with the default parameters of the Differential
    Evolution Algorithm.
    :rtype dict
    """
    return {'callback': None, 'max_iters': 10000, 'seed': None}


def de(population_size: int, individual_size: int, f: Union[float, int],
       cr: Union[float, int], bounds: np.ndarray,
       func: Callable[[np.ndarray], np.ndarray],
       callback: Callable[[Dict], Any],
       max_iters: int, seed: Union[int, None]) -> [np.ndarray, int]:
    """
    Applies the standard differential evolution algorithm.
    :param population_size: Size of the population.
    :type population_size: int
    :param individual_size: Number of gens/features of an individual.
    :type individual_size: int
    :param f: Mutation parameter. Must be in [0, 2].
    :type f: Union[float, int]
    :param cr: Crossover Ratio. Must be in [0, 1].
    :type cr: Union[float, int]
    :param bounds: Numpy ndarray with individual_size rows and 2 columns.
    First column represents the minimum value for the row feature.
    Second column represent the maximum value for the row feature.
    :type bounds: np.ndarray
    :param func: Evaluation function. The function used must receive one
     parameter.This parameter will be a numpy array representing an individual.
    :type func: Callable[[np.ndarray], float]
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param max_iters: Number of generations after the algorithm is stopped.
    :type max_iters: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]
    """
    # 0. Check parameters are valid
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer.")

    if type(individual_size) is not int or individual_size <= 0:
        raise ValueError("individual_size must be a positive integer.")

    if (type(f) is not int and type(f) is not float) or not 0 <= f <= 2:
        raise ValueError("f (mutation parameter) must be a "
                         "real number in [0,2].")

    if (type(cr) is not int and type(cr) is not float) or not 0 <= cr <= 1:
        raise ValueError("cr (crossover ratio) must be a "
                         "real number in [0,1].")

    if type(max_iters) is not int or max_iters <= 0:
        raise ValueError("max_iter must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    # 1. Initialization
    np.random.seed(seed)
    population = pyade.commons.init_population(population_size,
                                               individual_size, bounds)
    fitness = pyade.commons.apply_fitness(population, func)

    for num_iter in range(max_iters):
        mutated = pyade.commons.binary_mutation(population, f, bounds)
        crossed = pyade.commons.crossover(population, mutated, cr)
        c_fitness = pyade.commons.apply_fitness(crossed, func)
        population, indexes = pyade.commons.selection(population, crossed,
                                                      fitness, c_fitness, return_indexes=True)

        fitness[indexes] = c_fitness[indexes]
        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best]
