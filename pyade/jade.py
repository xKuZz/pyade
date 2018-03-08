import pyade.commons
import numpy as np
import scipy.stats
import random
from typing import Callable, Union


def get_default_jade_params():
    """
        Returns the default parameters of the JADE Differential Evolution Algorithm
        :return: Dict with the default parameters of the JADE Differential
        Evolution Algorithm.
        :rtype dict
        """
    return {'cr': 0.5, 'f': 0.5, 'max_iters': 10000, 'seed': None}


def jade(population_size: int, individual_size: int, f: Union[float, int],
         cr: Union[float, int], bounds: np.ndarray,
         func: Callable[[np.ndarray], float], p: Union[int, float], c: Union[int, float],
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
    :param p: Parameter to choose the best vectors. Must be in (0, 1].
    :type p: Union[int, float]
    :param c: Variable to control parameter adoption. Must be in [0, 1].
    :type c: Union[int, float]
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

    if type(p) not in [int, float] and 0 < p <= 1:
        raise ValueError("p must be a real number in (0, 1].")
    if type(p) not in [int, float] and 0 <= c <= 1:
        raise ValueError("c must be an rela number in [0, 1].")

    np.random.seed(seed)
    population = pyade.commons.init_population(population_size, individual_size, bounds)
    archive = []

    for num_iter in range(max_iters):
        fitness = pyade.commons.apply_fitness(population, func)
        s_f = []
        s_cr = []
        mutated = np.empty(population.shape)
        crossed = np.empty(population.shape)
        for i in range(population_size):
            # CR_i Normal Distribution
            cr_i = np.random.normal(cr, 0.1)
            cr_i = np.clip(cr_i, 0, 1)
            # F_i Cauchy Distribution
            f_i = 0
            while f_i == 0:
                f_i = scipy.stats.cauchy.rvs(loc=f, scale=0.1)
            if f_i >= 1:
                f_i = 1
            # Mutation
            # Find a random in the p best.
            options = np.argsort(fitness)[::-1][:round(p*population_size)]
            selected = np.random.choice(options)
            x_best = population[selected]
            # Find random from current population that is not the selected one
            selection_range = np.arange(0, population_size + len(archive))
            np.delete(selection_range, selected)
            selected_2 = np.random.choice(selection_range[:population_size-1])
            x_r1 = population[selected_2]
            np.delete(selection_range, selected_2 + 1)
            # Find random from current population or archive that wasn't selected previously
            selected_3 = np.random.choice(selection_range)
            if selected_3 < population_size:
                x_r2 = population[selected_3]
            else:
                x_r2 = archive[selected_3 - population_size - 2]

            mutated[i, :] = population[i, :] + f_i * (x_best - population[i, :]) + f_i * (x_r1 - x_r2)

            # Crossover
            jrand = np.random.randint(0, individual_size)
            for j in range(individual_size):
                if j == jrand or np.random.rand() < cr_i:
                    crossed[i, j] = mutated[i, j]
                else:
                    crossed[i, j] = population[i, j]

       # Replacement

            c_fitness = pyade.commons.apply_fitness([crossed[i]], func)
            if fitness[i] > c_fitness[0]:
                archive.append(population[i])
                s_cr.append(cr_i)
                s_f.append(f_i)
                population[i] = crossed[i]

        # Update crossover  and mutation control parameters.
        cr = (1 - c) * cr + c * np.mean(s_cr)
        f = (1 - c) * f + c * np.mean(s_f)

    if len(archive) > population_size:
        archive = random.sample(archive, population_size)

    fitness = pyade.commons.apply_fitness(population, func)
    best = np.argmin(fitness)
    return population[best], fitness[best]