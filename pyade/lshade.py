import pyade.commons
import numpy as np
import scipy.stats
import random
from typing import Callable, Union


def get_default_params():
    """
        Returns the default parameters of the L-SHADE Differential Evolution Algorithm
        :return: Dict with the default parameters of the JADE Differential
        Evolution Algorithm.
        :rtype dict
    """
    return {'max_iters': 10000, 'seed': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], np.ndarray],
          max_iters: int, seed: Union[int, None]) -> [np.ndarray, int]:
    """
    Applies the standard differential evolution algorithm.
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

    if type(max_iters) is not int or max_iters <= 0:
        raise ValueError("max_iter must be a positive integer.")

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
    init_size = population_size
    m_cr = np.ones(population_size) * 0.5
    m_f = np.ones(population_size) * 0.5
    archive = []
    k = 0
    fitness = pyade.commons.apply_fitness(population, func)

    all_indexes = list(range(population_size))
    for num_iter in range(max_iters):
        # 2.1 Adaptation
        r = np.random.choice(all_indexes, population_size)
        cr = np.random.normal(m_cr[r], 0.1, population_size)
        f = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=population_size)
        p = np.random.uniform(low=2/population_size, high=0.2)

        # 2.2 Common steps
        mutated = pyade.commons.current_to_pbest_mutation(population, fitness, f.reshape(len(f), 1), p, bounds)
        crossed = pyade.commons.crossover(population, mutated, cr.reshape(len(f), 1))
        c_fitness = pyade.commons.apply_fitness(crossed, func)
        population = pyade.commons.selection(population, crossed,
                                             fitness, c_fitness)





        # 2.3 Adapt for next generation
        indexes = np.where(c_fitness < fitness)[0]
        archive.extend(population[indexes])

        if len(indexes) > 0:
            if len(archive) > population_size:
                archive = random.sample(archive, population_size)

            weights = np.abs(fitness[indexes] - c_fitness[indexes])
            weights /= np.sum(weights)
            m_cr[k] = np.sum(weights * cr[indexes])
            m_f[k] = np.sum(f[indexes] ** 2) / np.sum(f[indexes])

            k += 1
            if k == population_size:
                k = 0

        fitness[indexes] = c_fitness[indexes]
        # Adapt population size
        new_population_size = round((4 - init_size) / max_iters * (num_iter + 1) + init_size)
        if population_size > new_population_size:
            population_size = new_population_size
            best_indexes = np.argsort(fitness)[:population_size]
            population = population[best_indexes]
            fitness = fitness[best_indexes]
            if k == population_size:
                k = 0
    best = np.argmin(fitness)
    return population[best], fitness[best]