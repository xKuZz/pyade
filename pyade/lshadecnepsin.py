import pyade.commons
import numpy as np
import math
import scipy.stats
import random
from typing import Callable, Union, Dict, Any


def get_default_params(dim: int):
    """
        Returns the default parameters of the L-SHADE-cnEpSin Differential Evolution Algorithm
        :param dim: Size of the problem (or individual).
        :type dim: int
        :return: Dict with the default parameters of the L-SHADE-cnEpSin  Differential
        Evolution Algorithm.
        :rtype dict
    """
    return {'population_size': 18 * dim,
            'min_population_size': 4,
            'individual_size': dim, 'memory_size': 5,
            'max_evals': 10000 * dim, 'seed': None, 'callback': None, 'opts': None,
            'terminate_callback': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          memory_size: int, callback: Callable[[Dict], Any],
          min_population_size: int,
          max_evals: int, seed: Union[int, None],
          terminate_callback: Callable[[], bool]) -> [np.ndarray, int]:
    """
    Applies the L-SHADE-cnEpSin differential evolution algorithm.
    :param population_size: Size of the population (NP-max)
    :type population_size: int
    :param min_population_size: Lowest size of the population (NP-min)
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
    # 1.1 Initialize population at first generation
    population = pyade.commons.init_population(population_size, individual_size, bounds)
    fitness = pyade.commons.apply_fitness(population, func, opts)

    # 1.2 Initialize memory of first control settings
    u_f = np.ones(memory_size) * .5
    u_cr = np.ones(memory_size) * .5
    # 1.3 Initialize memory of second control settings
    u_freq = np.ones(memory_size) * .5

    # 1.4 Initialize covariance matrix settings
    ps, pc = .5, .4

    lp = 20
    freq = 0.5
    p = 0.11

    current_size = population_size
    num_evals = population_size
    k = 0
    ns_1 = []
    nf_1 = []
    ns_2 = []
    nf_2 = []

    # Calculate max_iters
    n = population_size
    i = 0
    max_iters = 0
    if max_evals > 1e8:
        max_iters = 200 # Probably wrong, but what else should I do here? the following loop goes crazy if max_evals is too high
    else:
        while i < max_evals: # TODO causes problems with big values...
            max_iters += 1
            n = round((min_population_size - population_size) / max_evals * i + population_size)
            i += n

    current_generation = 0
    while num_evals < max_evals and (terminate_callback is not None and not terminate_callback()):
        # Mutation
        if current_generation <= (max_iters / 2):
            if current_generation <= lp:
                # 2.a Both sinusoidal configurations have the same probability
                p1 = 0.5
                p2 = 0.5
            else:
                success_option_1 = np.sum(ns_1) / (np.sum(ns_1) + np.sum(nf_1)) + 0.01
                success_option_2 = np.sum(ns_2) / (np.sum(ns_2) + np.sum(nf_2)) + 0.01

                p1 = success_option_1 / (success_option_1 + success_option_2)
                p2 = success_option_2 / (success_option_1 + success_option_2)

            option = np.random.choice(['p1', 'p2'], p=[p1, p2], size=current_size)
            p1_indexes = np.where(option == 'p1')[0]
            p2_indexes = np.where(option == 'p2')[0]
            f = np.empty(current_size)

            f[p1_indexes] = math.sin(2 * math.pi * freq * (current_generation + 1) + math.pi)
            f[p1_indexes] *= (max_iters - current_generation - 1) / max_iters
            f[p1_indexes] += 1
            f[p1_indexes] /= 2

            random_index = np.random.randint(0, memory_size)

            freq_i = np.empty(current_size)
            freq_i[p2_indexes] = scipy.stats.cauchy.rvs(loc=u_freq[random_index], scale=0.1, size=len(p2_indexes))

            f[p2_indexes] = np.sin(2 * math.pi * freq_i[p2_indexes] * (current_generation + 1))
            f[p2_indexes] *= (current_generation + 1) / max_iters
            f[p2_indexes] += 1
            f[p2_indexes] /= 2

            random_index = np.random.randint(0, memory_size)
            f = scipy.stats.cauchy.rvs(loc=u_f[random_index], scale=0.1, size=current_size)

            f[f > 1] = 0
            while sum(f <= 0) != 0:
                r = np.random.choice(list(range(memory_size)), sum(f <= 0))
                f[f <= 0] = scipy.stats.cauchy.rvs(loc=u_f[r], scale=0.1, size=sum(f <= 0))

            f = np.clip(f, 0.05, 1)
        else:
            random_index = np.random.randint(0, memory_size)
            f = scipy.stats.cauchy.rvs(loc=u_f[random_index], scale=0.1, size=current_size)

            f[f > 1] = 0
            while sum(f <= 0) != 0:
                r = np.random.choice(list(range(memory_size)), sum(f <= 0))
                f[f <= 0] = scipy.stats.cauchy.rvs(loc=u_f[r], scale=0.1, size=sum(f <= 0))

            f = np.clip(f, 0.05, 1)

        mutated = pyade.commons.current_to_pbest_mutation(population, fitness, f.reshape(current_size, 1),
                                                          np.ones(current_size) * p, bounds)

        # Crossover
        random_index = np.random.randint(0, memory_size)
        cr = np.random.normal(loc=u_cr[random_index], scale=0.1, size=current_size)
        cr = np.clip(cr, 0, 1)
        cr[u_cr[random_index] == 1] = 0
        cr[cr == 1] = 0
        randoms = np.random.rand(current_size)
        cov_indexes = np.where(randoms < pc)[0]
        bin_indexes = np.where(randoms >= pc)[0]

        crossed = population.copy()
        # Crossover: Binomial Crossover
        crossed[bin_indexes] = pyade.commons.crossover(population[bin_indexes], mutated[bin_indexes],
                                                       cr[bin_indexes].reshape(len(bin_indexes), 1))

        # Covariance matrix learning with euclidean neighborhood
        # A. Search for best in population
        best_index = np.argsort(fitness)
        best = population[best_index]

        # B. Compute euclidean distances between best and the rest of the population
        distances = np.linalg.norm(population - best, axis=1)
        indexes = np.argsort(distances)[:round(ps * current_size)]

        # C. Compute covariance matrix and its matrix decomposition
        xsel = population[indexes]
        sel = round(ps * current_size)
        xmean = np.mean(xsel, axis=1)

        aux = np.ones((sel, 1), dtype=bool)
        xsel = xsel.T
        c = 1 / (sel - 1) * np.dot(xsel - xmean, (xsel - xmean).T)
        c = np.triu(c) + np.triu(c, 1).T
        r, d = np.linalg.eig(c)
        if np.max(np.diag(d)) > 1e20 * np.min(np.diag(d)):
            tmp = np.max(np.diag(d))/1e20 - np.min(np.diag(d))
            c = c + tmp * np.eye(individual_size)
            r, d = np.linalg.eig(c)

        tm = d
        tm_ = d.T

        # D. Apply coordinate origin transform

        cov_population = np.dot(population[cov_indexes], tm)
        cov_mutated = np.dot(mutated[cov_indexes], tm)

        cov_crossed = pyade.commons.crossover(cov_population, cov_mutated,
                                              cr[cov_indexes].reshape(len(cov_indexes), 1))

        # E. Go back the te original coordinate system

        crossed[cov_indexes] = np.dot(cov_crossed, tm_.T)
        crossed_fitness = pyade.commons.apply_fitness(crossed, func, opts)
        num_evals += current_size

        # Selection
        population, indexes = pyade.commons.selection(population, crossed, fitness,
                                                      crossed_fitness, return_indexes=True)
        winners = crossed_fitness < fitness

        # Update success lists to recalculate probabilities
        if current_generation <= (max_iters / 2):
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
            if max(cr[indexes]) != 0:
                u_cr[k] = np.sum(weights * cr[indexes] ** 2) / np.sum(weights * cr[indexes])

            if current_generation > (max_iters / 2):
                u_f[k] = np.sum(weights * f[indexes] ** 2) / np.sum(weights * f[indexes])

            if lp < current_generation < (max_iters / 2):
                chosen = np.logical_and(np.array(option == 'p2', dtype=bool), winners)
                if len(freq_i[chosen]) != 0:
                    u_freq[k] = np.mean(freq_i[chosen])
                if np.isnan(u_freq[k]):
                    u_freq[k] = 0.5

            k += 1
            if k == memory_size:
                k = 0

        fitness[indexes] = crossed_fitness[indexes]

        # Linear population reduction
        new_population_size = round((min_population_size - population_size) / max_evals * num_evals + population_size)
        if current_size > new_population_size:
            current_size = new_population_size
            best_indexes = np.argsort(fitness)[:current_size]
            population = population[best_indexes]
            fitness = fitness[best_indexes]
            if k == memory_size:
                k = 0

        if callback is not None:
            callback(**(locals()))

        current_generation += 1

    best = np.argmin(fitness)
    return population[best], fitness[best]
