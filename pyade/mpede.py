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
            'ng': 20, 'c': 0.1, 'p': 0.04
            }


#TODO: REVISAR DOCUMENTACIÓN
def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], float],
          callback: Callable[[Dict], Any],
          lambdas: Union[list, np.array],
          ng: int, c: Union[int, float], p: Union[int, float],
          max_evals: int, seed: Union[int, None]) -> [np.ndarray, int]:

    """
    Applies the MPEDE differential evolution algorithm.
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
    :param memory_size: Size of the internal memory.
    :type memory_size: int
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param max_evals: Number of evaluations after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]
    """

    # 0. Check external parameters


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
        fitnesses.append(pyade.commons.apply_fitness(pops[j], func))
        num_evals += len(pops[j])

    # 2. Start the algorithm
    while num_evals <= max_evals:
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
        crossed2 = pyade.commons.crossover(pops[1], mutated2, cr[1].reshape(len(cr[1]), 1))
        crossed3 = pyade.commons.crossover(pops[2], mutated3, cr[2].reshape(len(cr[2]), 1))

        c_fitness1 = pyade.commons.apply_fitness(crossed1, func)
        c_fitness2 = pyade.commons.apply_fitness(crossed2, func)
        c_fitness3 = pyade.commons.apply_fitness(crossed3, func)

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