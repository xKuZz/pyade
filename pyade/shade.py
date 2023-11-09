import pyade.commons
import numpy as np
import scipy.stats
import random
from typing import Callable, Union, Dict, Any


def get_default_params(dim: int):
    """
        Returns the default parameters of the SHADE Differential Evolution Algorithm.
        :param dim: Size of the problem (or individual).
        :type dim: int
        :return: Dict with the default parameters of the SHADE Differential
        Evolution Algorithm.
        :rtype dict
    """
    return {'max_evals': 10000 * dim, 'memory_size': 100,
            'individual_size': dim, 'population_size': 10 * dim,
            'callback': None, 'seed': None, 'opts': None,
            'terminate_callback': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          memory_size: int, callback: Callable[[Dict], Any],
          max_evals: int, seed: Union[int, None],
          terminate_callback: Callable[[], bool]) -> [np.ndarray, int]:
    """
    Applies the SHADE differential evolution algorithm.
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
    :param opts: Optional parameters for the fitness function.
    :type opts: Any type.
    :param memory_size: Size of the internal memory.
    :type memory_size: int
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
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

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    np.random.seed(seed)
    random.seed(seed)

    # 1. Initialization
    population = pyade.commons.init_population(population_size, individual_size, bounds)
    m_cr = np.ones(memory_size) * 0.5
    m_f = np.ones(memory_size) * 0.5
    archive = []
    k = 0
    fitness = pyade.commons.apply_fitness(population, func, opts)

    all_indexes = list(range(memory_size))
    max_iters = max_evals // population_size
    for current_generation in range(max_iters):
        if terminate_callback is not None and terminate_callback():
            break
        # 2.1 Adaptation
        r = np.random.choice(all_indexes, population_size)
        cr = np.random.normal(m_cr[r], 0.1, population_size)
        cr = np.clip(cr, 0, 1)
        cr[cr == 1] = 0
        f = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=population_size)
        f[f > 1] = 0

        while sum(f <= 0) != 0:
            r = np.random.choice(all_indexes, sum(f <= 0))
            f[f <= 0] = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=sum(f <= 0))

        p = np.random.uniform(low=2/population_size, high=0.2, size=population_size)

        # 2.2 Common steps
        mutated = pyade.commons.current_to_pbest_mutation(population, fitness, f.reshape(len(f), 1), p, bounds)
        crossed = pyade.commons.crossover(population, mutated, cr.reshape(len(f), 1))
        c_fitness = pyade.commons.apply_fitness(crossed, func, opts)
        population, indexes = pyade.commons.selection(population, crossed,
                                                      fitness, c_fitness, return_indexes=True)

        # 2.3 Adapt for next generation
        archive.extend(population[indexes])

        if len(indexes) > 0:
            if len(archive) > memory_size:
                archive = random.sample(archive, memory_size)
            if max(cr) != 0:
                weights = np.abs(fitness[indexes] - c_fitness[indexes])
                weights = weights.astype(float)
                weights /= np.sum(weights)
                m_cr[k] = np.sum(weights * cr[indexes])
            else:
                m_cr[k] = 1

            m_f[k] = np.sum(f[indexes]**2)/np.sum(f[indexes])

            k += 1
            if k == memory_size:
                k = 0

        fitness[indexes] = c_fitness[indexes]
        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best]
