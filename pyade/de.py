import numpy as np
import pyade.commons
from typing import Callable, Union, Dict, Any


def get_default_params(dim: int) -> dict:
    """
    Returns the default parameters of the Differential Evolution Algorithm
    :param dim: Size of the problem (or individual).
    :type dim: int
    :return: Dict with the default parameters of the Differential
    Evolution Algorithm.
    :rtype dict
    """
    return {'callback': None, 'max_evals': 10000 * dim, 'seed': None, 'cross': 'bin',
            'f': 0.5, 'cr': 0.9, 'individual_size': dim, 'population_size': 10 * dim, 'opts': None,
            'terminate_callback': None}


def apply(population_size: int, individual_size: int, f: Union[float, int],
          cr: Union[float, int], bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          callback: Callable[[Dict], Any],
          cross: str,
          max_evals: int, seed: Union[int, None],
          terminate_callback: Callable[[], bool]) -> [np.ndarray, int]:
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
    :param opts: Optional parameters for the fitness function.
    :type opts: Any type.
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param cross: Indicates whether to use the binary crossover('bin') or the exponential crossover('exp').
    :type cross: str
    :param max_evals: Number of evaluations after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :param terminate_callback: Callback that checks whether it is time to terminate or not. The callback should return True if it's time to stop, otherwise False.
    :type terminate_callback: Callable[[], bool]
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

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(cross) is not str and cross not in ['bin', 'exp']:
        raise ValueError("cross must be a string and must be one of \'bin\' or \'cross\'")
    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    # 1. Initialization
    np.random.seed(seed)
    population = pyade.commons.init_population(population_size,
                                               individual_size, bounds)
    fitness = pyade.commons.apply_fitness(population, func, opts)

    max_iters = max_evals // population_size
    for current_generation in range(max_iters):
        if terminate_callback is not None and terminate_callback():
            break

        mutated = pyade.commons.binary_mutation(population, f, bounds)
        if cross == 'bin':
            crossed = pyade.commons.crossover(population, mutated, cr)
        else:
            crossed = pyade.commons.exponential_crossover(population, mutated, cr)

        c_fitness = pyade.commons.apply_fitness(crossed, func, opts)
        population, indexes = pyade.commons.selection(population, crossed,
                                                      fitness, c_fitness, return_indexes=True)

        fitness[indexes] = c_fitness[indexes]
        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best]
