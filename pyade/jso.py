import pyade.commons
import numpy as np
import scipy.stats
import random
from typing import Callable, Union, Dict, Any


def get_default_params(dim: int):
    """
        Returns the default parameters of the jSO Algorithm
        :param dim: Size of the problem (or individual).
        :type dim: int
        :return: Dict with the default parameters of the jSO Algorithm.
        :rtype dict
    """
    return {'population_size': int(round(25 * np.log(dim) * np.sqrt(dim))),
            'individual_size': dim, 'memory_size': 5,
            'max_evals': 10000 * dim, 'seed': None, 'callback': None, 'opts': None,
            'terminate_callback': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          memory_size: int, callback: Callable[[Dict], Any],
          max_evals: int, seed: Union[int, None],
          terminate_callback: Callable[[], bool]) -> [np.ndarray, int]:
    """
    Applies the jSO differential evolution algorithm.
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
    current_size = population_size
    m_cr = np.ones(population_size) * .8
    m_f = np.ones(population_size) * .5
    archive = []
    k = 0
    fitness = pyade.commons.apply_fitness(population, func, opts)

    memory_size = population_size
    memory_indexes = list(range(memory_size))
    num_evals = population_size
    current_generation = 0
    p_max = .25
    p_min = p_max / 2
    p = p_max

    # Calculate max_iters
    n = population_size
    i = 0
    max_iters = 0
    while i < max_evals:
        max_iters += 1
        n = round((4 - population_size) / max_evals * i + population_size)
        i += n

    while num_evals < max_evals and (terminate_callback is not None and not terminate_callback()):
        # 2.1 Adaptation
        r = np.random.choice(memory_indexes, current_size)
        m_cr[- 1] = 0.9
        m_f[-1] = 0.9

        cr = np.random.normal(m_cr[r], 0.1, current_size)
        cr = np.clip(cr, 0, 1)
        cr[m_cr[r] == 1] = 0
        cr[m_cr[r] < 0] = 0

        if current_generation < (max_iters / 4):
            cr[cr < 0.7] = 0.7
        elif current_generation < (max_iters / 2):
            cr[cr < 0.6] = 0.6

        f = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=current_size)
        while sum(f <= 0) != 0:
            r = np.random.choice(memory_indexes, sum(f <= 0))
            f[f <= 0] = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=sum(f <= 0))

        f = np.clip(f, 0, 1)
        if current_generation < 0.6 * max_iters:
            f = np.clip(f, 0, 0.7)


        # 2.2 Common steps
        # 2.2.1 Calculate weights for mutation
        weighted = f.copy().reshape(len(f), 1)

        if num_evals < 0.2 * max_evals:
            weighted *= .7
        elif num_evals < 0.4 * max_evals:
            weighted *= .8
        else:
            weighted *= 1.2

        weighted = np.clip(weighted, 0, 1)
        # print(min(fitness), min(cr), max(cr), min(f), max(f))
        mutated = pyade.commons.current_to_pbest_weighted_mutation(population, fitness, f.reshape(len(f), 1),
                                                                   weighted, p, bounds)
        crossed = pyade.commons.crossover(population, mutated, cr.reshape(len(f), 1))
        c_fitness = pyade.commons.apply_fitness(crossed, func, opts)

        num_evals += current_size
        population, indexes = pyade.commons.selection(population, crossed,
                                                      fitness, c_fitness, return_indexes=True)

        # 2.3 Adapt for next generation
        archive.extend(population[indexes])

        if len(indexes) > 0:
            if len(archive) > population_size:
                archive = random.sample(archive, population_size)

            weights = np.abs(fitness[indexes] - c_fitness[indexes])
            weights /= np.sum(weights)

            if max(cr) != 0:
                m_cr[k] = (np.sum(weights * cr[indexes]**2) / np.sum(weights * cr[indexes]) + m_cr[-1]) / 2
            else:
                m_cr[k] = 1

            m_f[k] = np.sum(weights * f[indexes]**2) / np.sum(weights * f[indexes])

            k += 1
            if k == memory_size:
                k = 0

        fitness[indexes] = c_fitness[indexes]
        # Adapt population size
        new_population_size = round((4 - population_size) / max_evals * num_evals + population_size)
        if current_size > new_population_size:
            current_size = new_population_size
            best_indexes = np.argsort(fitness)[:current_size]
            population = population[best_indexes]
            fitness = fitness[best_indexes]
            if k == memory_size:
                k = 0

        # Adapt p
        p = (p_max - p_min) / max_evals * num_evals + p_min
        if callback is not None:
            callback(**(locals()))

        current_generation += 1

    best = np.argmin(fitness)
    return population[best], fitness[best]
