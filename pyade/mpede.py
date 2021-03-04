from typing import Any, Callable, Dict, Union
import numpy as np
import pyade.commons
import scipy.stats


def get_default_params(dim: int) -> dict:
    """
        Returns the default parameters of the Multi-population ensemble DE (MPEDE)
        :param dim: Size of the problem (or individual).
        :type dim: int
        :return: Dict with the default parameters of the MPEDE Differential
        Evolution Algorithm.
        :rtype dict
        """
    pop_size = 250
    return {'max_evals': 10000 * dim, 'individual_size': dim, 'callback': None,
            'population_size': pop_size, 'seed': None, 'lambdas': [0.2, 0.2, 0.2, 0.4],
            'ng': 20, 'c': 0.1, 'p': 0.04, 'opts': None,
            'terminate_callback': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          callback: Callable[[Dict], Any],
          lambdas: Union[list, np.array],
          ng: int, c: Union[int, float], p: Union[int, float],
          max_evals: int, seed: Union[int, None],
          terminate_callback: Callable[[], bool]) -> [np.ndarray, int]:

    """
    Applies the MPEDE differential evolution algorithm.
    :param population_size: Size of the population (NP-max)
    :type population_size: int
    :param ng: Number of generations after the best strategy is updated.
    :type ng: int
    :param lambdas: Percentages of each of the 4 subpopulations.
    :type lambdas: Union[list, np.array]
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
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param max_evals: Number of evaluations after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :param p: Parameter to choose the best vectors. Must be in (0, 1].
    :type p: Union[int, float]
    :param c: Variable to control parameter adoption. Must be in [0, 1].
    :type c: Union[int, float]
    :type seed: Union[int, None]
    :param terminate_callback: Callback that checks whether it is time to terminate or not. The callback should return True if it's time to stop, otherwise False.
    :type terminate_callback: Callable[[], bool]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]

    """

    # 0. Check external parameters
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

    if type(p) not in [int, float] and 0 < p <= 1:
        raise ValueError("p must be a real number in (0, 1].")

    if type(c) not in [int, float] and 0 <= c <= 1:
        raise ValueError("c must be an real number in [0, 1].")

    if type(ng) is not int:
        raise ValueError("ng must be a positive integer number.")

    if type(lambdas) not in [list, np.ndarray] and len(lambdas) != 4 and sum(lambdas) != 1:
        raise ValueError("lambdas must be a list or npdarray of 4 numbers that sum 1.")

    np.random.seed(seed)

    # 1. Initialize internal parameters
    # 1.1 Control parameters
    u_cr = np.ones(3) * 0.5
    u_f = np.ones(3) * 0.5
    f_var = np.zeros(3)
    fes = np.zeros(3)

    # 1.2 Initialize population
    pop_size = lambdas * population_size
    big_population = pyade.commons.init_population(int(sum(pop_size)), individual_size, bounds)
    pops = np.array_split(big_population, 4)

    chosen = np.random.randint(0, 3)
    newpop = np.concatenate((pops[chosen], pops[3]))
    pops[chosen] = newpop
    pop_size = list(map(len, pops))
    current_generation = 0
    num_evals = 0

    f = []
    cr = []
    fitnesses = []
    for j in range(3):
        f.append(np.empty(pop_size[j]))
        cr.append(np.empty(pop_size[j]))
        fitnesses.append(pyade.commons.apply_fitness(pops[j], func, opts))
        num_evals += len(pops[j])

    # 2. Start the algorithm
    while num_evals < max_evals and (terminate_callback is not None and not terminate_callback()):
        current_generation += 1

        # 2.1 Generate CR and F values
        for j in range(3):
            f[j] = scipy.stats.cauchy.rvs(loc=u_f[j], scale=0.1, size=len(pops[j]))
            f[j] = np.clip(f[j], 0, 1)

            cr[j] = np.random.normal(u_cr[j], 0.1, len(pops[j]))
            cr[j] = np.clip(cr[j], 0, 1)

        # 2.2 Apply mutation to each subpopulation
        mutated1 = pyade.commons.current_to_pbest_mutation(pops[0], fitnesses[0],
                                                           f[0].reshape(len(f[0]), 1),
                                                           np.ones(len(pops[0])) * p, bounds)

        mutated2 = pyade.commons.current_to_rand_1_mutation(pops[1], fitnesses[1],
                                                            f[1].copy().reshape(len(f[1]), 1) * .5 + 1,
                                                            f[1].reshape(len(f[1]), 1), bounds)

        mutated3 = pyade.commons.binary_mutation(pops[2], f[2].reshape(len(f[2]), 1), bounds)

        # 2.3 Do the crossover and calculate new fitness
        crossed1 = pyade.commons.crossover(pops[0], mutated1, cr[0].reshape(len(cr[0]), 1))
        crossed2 = mutated2
        crossed3 = pyade.commons.crossover(pops[2], mutated3, cr[2].reshape(len(cr[2]), 1))

        c_fitness1 = pyade.commons.apply_fitness(crossed1, func, opts)
        c_fitness2 = pyade.commons.apply_fitness(crossed2, func, opts)
        c_fitness3 = pyade.commons.apply_fitness(crossed3, func, opts)

        for j in range(3):
            num_evals += len(pops[j])
            fes[j] += len(pops[j])

        # 2.4 Do the selection and update control parameters
        winners1 = c_fitness1 < fitnesses[0]
        winners2 = c_fitness2 < fitnesses[1]
        winners3 = c_fitness3 < fitnesses[2]

        pops[0] = pyade.commons.selection(pops[0], crossed1, fitnesses[0], c_fitness1)
        pops[1] = pyade.commons.selection(pops[1], crossed2, fitnesses[1], c_fitness2)
        pops[2] = pyade.commons.selection(pops[2], crossed3, fitnesses[2], c_fitness3)

        fitnesses[0][winners1] = c_fitness1[winners1]
        fitnesses[1][winners2] = c_fitness2[winners2]
        fitnesses[2][winners3] = c_fitness3[winners3]

        if sum(winners1) != 0 and np.sum(f[0][winners1]) != 0:
            u_cr[0] = (1 - c) * u_cr[0] + c * np.mean(cr[0][winners1])
            u_f[0] = (1 - c) * u_f[0] + c * (np.sum(f[0][winners1] ** 2) / np.sum(f[0][winners1]))
        if sum(winners2) != 0 and np.sum(f[1][winners2]) != 0:
            u_cr[1] = (1 - c) * u_cr[1] + c * np.mean(cr[1][winners2])
            u_f[1] = (1 - c) * u_f[1] + c * (np.sum(f[1][winners2] ** 2) / np.sum(f[1][winners2]))
        if sum(winners3) != 0 and np.sum(f[2][winners3]) != 0:
            u_cr[2] = (1 - c) * u_cr[2] + c * np.mean(cr[2][winners3])
            u_f[2] = (1 - c) * u_f[2] + c * (np.sum(f[2][winners3] ** 2) / np.sum(f[2][winners3]))

        fes[0] += np.sum(fitnesses[0][winners1] - c_fitness1[winners1])
        fes[1] += np.sum(fitnesses[1][winners2] - c_fitness2[winners2])
        fes[2] += np.sum(fitnesses[2][winners3] - c_fitness3[winners3])

        population = np.concatenate((pops[0], pops[1], pops[2]))
        fitness = np.concatenate((fitnesses[0], fitnesses[1], fitnesses[2]))

        if current_generation % ng == 0:
            k = [f_var[i] / len(pops[i] / ng) for i in range(3)]
            chosen = np.argmax(k)

        indexes = np.arange(0, len(population), 1, np.int)
        np.random.shuffle(indexes)
        indexes = np.array_split(indexes, 4)
        chosen = np.random.randint(0, 3)

        pops = []
        fitnesses = []
        f = []
        cr = []

        for j in range(3):

            if j == chosen:
                pops.append(np.concatenate((population[indexes[j]], population[indexes[3]])))
                fitnesses.append(np.concatenate((fitness[indexes[j]], fitness[indexes[3]])))
            else:
                pops.append(population[indexes[j]])
                fitnesses.append(fitness[indexes[j]])

            f.append(np.empty(len(pops[j])))
            cr.append(np.empty(len(pops[j])))

        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best]
