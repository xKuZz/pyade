import pyade.commons
import numpy as np
import math
import scipy.stats
import numpy.linalg
import random
from typing import Callable, Union


def get_default_params():
    """
        Returns the default parameters of the L-SHADE-cnEpSin Differential Evolution Algorithm
        :return: Dict with the default parameters of the L-SHADE-cnEpSin  Differential
        Evolution Algorithm.
        :rtype dict
    """
    return {'max_iters': 10000, 'seed': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], np.ndarray],
          max_iters: int, seed: Union[int, None]) -> [np.ndarray, int]:
    """
    Applies the L-SHADE-cnEpSin differential evolution algorithm.
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
    # 1.1 Initialize population at first generation
    population = pyade.commons.init_population(population_size, individual_size, bounds)
    fitness = pyade.commons.apply_fitness(population, func)

    # 1.2 Initialize memory of first control settings
    memory_size = 5
    u_f = np.ones(memory_size) * .5
    u_cr = np.ones(memory_size) * .5
    # 1.3 Initialize memory of second control settings
    u_freq = .5

    # 1.4 Initialize covariance matrix settings
    ps, pc = .5, .4

    lp = 20
    freq = 0.5
    p = 0.11
    freq_i = np.ones(population_size) * 0.5

    current_size = population_size
    evals = population_size
    k = 0
    ns_1 = []
    nf_1 = []
    ns_2 = []
    nf_2 = []

    max_evals = max_iters * population_size
    for num_iter in range(max_iters):
        if evals >= max_evals:
            break
        # Mutation
        if num_iter <= (max_iters / 2):
            if num_iter <= lp:
                # 2.a Both sinusoidal configurations have the same probability
                p1 = 0.5
                p2 = 0.5
            else:
                success_option_1 = np.sum(ns_1) / (np.sum(ns_1) + np.sum(nf_1)) + 0.01
                success_option_2 = np.sum(ns_2) / (np.sum(ns_2) + np.sum(nf_2)) + 0.01

                p1 = success_option_1 / (success_option_1 + success_option_2)
                p2 = success_option_2 / (success_option_1 + success_option_2)
                pass

            option = np.random.choice(['p1', 'p2'], p=[p1, p2], size= current_size)
            p1_indexes = np.where(option == 'p1')[0]
            p2_indexes = np.where(option == 'p2')[0]
            f = np.empty(current_size)

            f[p1_indexes] = math.sin(2 * math.pi * freq * num_iter + math.pi)
            f[p1_indexes] *= (max_iters - num_iter) / max_iters
            f[p1_indexes] += 1
            f[p1_indexes] /= 2

            f[p2_indexes] = np.sin(2 * math.pi * freq_i[p2_indexes] * num_iter)
            f[p2_indexes] *= num_iter / max_iters
            f[p2_indexes] /= 2

        else:
            random_index = np.random.randint(0, memory_size)
            f = scipy.stats.cauchy.rvs(loc=u_f[random_index], scale=0.1, size=current_size)

        mutated = pyade.commons.current_to_pbest_mutation(population, fitness, f.reshape(current_size, 1),
                                                              p, bounds)


        # Crossover
        random_index = np.random.randint(0, memory_size)
        cr = np.random.normal(loc=u_cr[random_index], scale=0.1, size=current_size)
        randoms = np.random.rand(current_size)
        cov_indexes = np.where(randoms < pc)[0]
        bin_indexes = np.where(randoms > pc)[0]

        crossed = np.empty(population.shape)
        # Covariance matrix learning with euclidean neighborhood
        # A. Search for best in population
        best_index = np.argsort(fitness)
        best = population[best_index]

        # B. Compute euclidean distances between best and the rest of the population
        distances = np.linalg.norm(population - best, axis=1)
        indexes = np.argsort(distances)[:round(ps * current_size)]

        # C. Compute covariance matrix and its matrix decomposition

        covariance_matrix = np.cov(population[indexes], rowvar=False)

        b, d, b_t = np.linalg.svd(covariance_matrix)
        b = np.matrix(b)
        b_t = np.matrix(b_t)

        # D. Apply coordinate origin transform
        cov_population = np.empty(population.shape)
        for index in cov_indexes:
            cov_population[index] = np.array(b_t * np.matrix(population[index]).T).T
            mutated[index] = np.array(b_t * np.matrix(mutated[index]).T).T

        crossed[cov_indexes] = pyade.commons.crossover(population[cov_indexes], mutated[cov_indexes],
                                                       cr[cov_indexes].reshape(len(cov_indexes), 1))
        # E. Go back the te original coordinate system
        for index in cov_indexes:
            crossed[index] = np.array(b * np.matrix(crossed[index]).T).T

        # Crossover: Binomial Crossover
        crossed[bin_indexes] = pyade.commons.crossover(population[bin_indexes], mutated[bin_indexes],
                                                       cr[bin_indexes].reshape(len(bin_indexes), 1))

        crossed_fitness = pyade.commons.apply_fitness(crossed, func)

        # Selection
        population, indexes = pyade.commons.selection(population, crossed, fitness, crossed_fitness, return_indexes=True)
        winners = crossed_fitness < fitness


        # Update success lists to recalculate probabilities
        if num_iter <= (max_iters / 2):
            if len(ns_1) == lp:
                del ns_1[0]
                del ns_2[0]
                del nf_1[0]
                del nf_2[0]

            ns_1.append(np.sum(np.bitwise_and(winners, option == 'p1')))
            ns_2.append(np.sum(np.bitwise_and(winners, option == 'p2')))
            nf_1.append(np.sum(np.bitwise_and(np.bitwise_not(winners), option == 'p1')))
            nf_2.append(np.sum(np.bitwise_and(np.bitwise_not(winners), option == 'p2')))


        # Update memory
        if len(indexes) > 0:
            weights = np.abs(fitness[indexes] - crossed_fitness[indexes])
            weights /= np.sum(weights)
            u_cr[k] = np.sum(weights * cr[indexes])
            if num_iter > (max_iters / 2):
                u_f[k] = np.sum(f[indexes] ** 2) / np.sum(f[indexes])

            k += 1
            if k == memory_size:
                k = 0

        fitness[indexes] = crossed_fitness[indexes]

        # Linear population reduction
        new_population_size = round((4 - population_size) / max_evals * (evals + 1) + population_size)
        if population_size > new_population_size:
            population_size = new_population_size
            best_indexes = np.argsort(fitness)[:population_size]
            population = population[best_indexes]
            fitness = fitness[best_indexes]
            freq_i = freq_i[best_indexes]
            if k == memory_size:
                k = 0

    best = np.argmin(fitness)
    return population[best], fitness[best]


