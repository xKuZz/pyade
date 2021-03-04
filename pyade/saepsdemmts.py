import numpy as np
import pyade.commons
import collections
import pyade.mmts
from typing import Union, Callable, Dict, Any


def clearing(sigma, kappa, population, fitness):
    def __my_distance(a, b):
        return abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    for i in range(len(fitness)):
        if fitness[i] > 0:
            nbWinners = 1
            for j in range(1, len(fitness)):
                if fitness[j] > 0 and  __my_distance(population[i], population[j]) < sigma:
                    if nbWinners < kappa:
                        nbWinners += 1
                    else:
                        fitness[j] = 0

    return np.argsort(fitness)[:kappa]


def get_default_params(dim: int) -> dict:
    """
    Returns the default parameters of the Self-adaptive Differential Evolution Algorithm (SaDE).
    :param dim: Size of the problem (or individual).
    :type dim: int
    :return: Dict with the default parameters of SaDe
    :rtype dict
    """
    return {'max_evals': 10000 * dim, 'population_size': 60, 'callback': None,
            'individual_size': dim, 'seed': None, 'opts': None,
            'terminate_callback': None}


def apply(population_size: int, individual_size: int,
          bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          callback: Callable[[Dict], Any],
          max_evals: int, seed: Union[int, None],
          terminate_callback: Callable[[], bool]) -> [np.ndarray, int]:
    """
    Applies the Self-adaptive differential evolution algorithm (SaDE).
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
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param max_evals: Number of evaluatios after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]
    """

    np.random.seed(seed)
    population = pyade.commons.init_population(population_size, individual_size, bounds)

    fitness = pyade.commons.apply_fitness(population, func, opts)

    num_evals = 0

    pf1, pf2, pf3 = 1 / 3, 1 / 3, 1 / 3
    cr1, cr2, cr3 = 1 / 3, 1 / 3, 1 / 3
    bin_cross = 0.5
    best_mutation = 0.5

    current_generation = 1
    learning_period = 50
    update = True

    s = collections.defaultdict(lambda: np.zeros(learning_period))

    learning_k = 0
    run_mmts = True
    mmts_desired_evals = 60
    num_no_mmts = 0

    while num_evals < max_evals and (terminate_callback is not None and not terminate_callback()):

        # 1. Generate ensemble parameters
        if current_generation == 1 or current_generation > learning_period:
            f = np.random.choice([0.3, 0.5, 0.7], p=[pf1, pf2, pf3], size=population_size)
            cr = np.random.choice([0.1, 0.5, 0.9], p=[cr1, cr2, cr3], size=population_size)
            mutation_strat = np.random.choice(['rand', 'best'], p=[1 - best_mutation, best_mutation], size=population_size)
            cross_method = np.random.choice(['bin', 'exp'], p=[bin_cross, 1 - bin_cross], size=population_size)

        # 2.0 Niching
        m = 5
        pop_b = population.copy().reshape(population.shape[0], 1, population.shape[1])
        distances = np.sqrt(np.einsum('ijk, ijk->ij', population-pop_b, population-pop_b))
        neighbors = np.argsort(distances, axis=1)
        l_best_indexes = neighbors[:, 1]
        neighbors = neighbors[l_best_indexes, 1:m]

        # 2.1 Mutation
        rand_mut_idx = np.where(mutation_strat == 'rand')[0]
        best_mut_idx = np.where(mutation_strat == 'best')[0]
        mutated = np.empty(population.shape)

        # 2.1.a Rand mutation
        # Generate parents indexes
        choices = np.arange(0, neighbors.shape[1])
        parents1 = np.empty(population.shape)
        parents2 = np.empty(population.shape)
        parents3 = np.empty(population.shape)

        for i in range(population_size):
            choice = np.random.choice(choices, 3, replace=False)
            parents1[i] = population[neighbors[i, choice[0]]]
            parents2[i] = population[neighbors[i, choice[1]]]
            parents3[i] = population[neighbors[i, choice[2]]]

        mutated[rand_mut_idx] = parents1[rand_mut_idx] + f[rand_mut_idx].reshape(len(rand_mut_idx), 1) * (parents2[rand_mut_idx] - parents3[rand_mut_idx])

        mutated[best_mut_idx] = population[l_best_indexes[best_mut_idx]] + f[best_mut_idx].reshape(len(best_mut_idx), 1) * (parents1[best_mut_idx] - parents2[best_mut_idx])

        mutated = pyade.commons.keep_bounds(mutated, bounds)

        # 2.2 Crossover
        bin_cross_idx = np.where(cross_method == 'bin')[0]
        exp_cross_idx = np.where(cross_method == 'exp')[0]

        crossed = np.empty(population.shape)
        crossed[bin_cross_idx] = pyade.commons.crossover(population[bin_cross_idx], mutated[bin_cross_idx],
                                                         cr[bin_cross_idx].reshape(len(cr[bin_cross_idx]), 1))
        crossed[exp_cross_idx] = pyade.commons.exponential_crossover(population[exp_cross_idx], mutated[exp_cross_idx],
                                                                     cr[exp_cross_idx].reshape(len(cr[exp_cross_idx]), 1))

        # 2.3 Recalculate fitness
        c_fitness = pyade.commons.apply_fitness(crossed, func, opts)
        num_evals += population_size

        # 2.4 Distance between new population and original population
        distances = np.sqrt(np.einsum('ijk, ijk->ij', crossed - pop_b, crossed - pop_b))
        neighbors = np.argsort(distances, axis=1)
        l_best_indexes = neighbors[:, 1]

        selection = [c_fitness[i] < fitness[l_best_indexes[i]] for i in range(len(population))]
        population[l_best_indexes[selection]] = crossed[selection]
        fitness[l_best_indexes[selection]] = c_fitness[selection]

        if 1 < current_generation < learning_period:
            num_no_mmts += 60
            if num_no_mmts >= num_evals_mmts:
                run_mmts = True
                num_no_mmts = 0

        if current_generation > learning_period:
            if np.random.randn() > p_mmts:
                run_mmts = True

        if run_mmts:
            selected = clearing(0.2, 5, population, fitness.copy())
            a = pyade.mmts.mmts(population[selected], bounds, fitness[selected], mmts_desired_evals, func, opts)
            population[selected] = a[0]
            fitness_test = fitness[selected] < a[1]
            s['mmts_ok'] = len(fitness_test)
            s['mmts_fail'] = 5 - len(fitness_test)
            fitness[selected] = a[1]
            num_evals += a[2]

            if current_generation < learning_period:
                num_evals_mmts = a[2]

            run_mmts = False

        # 3. Update control parameters
        s['pf1_ok'][learning_k] = sum(np.logical_and(f == 0.3, selection))
        s['pf1_fail'][learning_k] = sum(f == 0.3) - s['pf1_ok'][learning_k]
        s['pf2_ok'][learning_k] = sum(np.logical_and(f == 0.5, selection))
        s['pf2_fail'][learning_k] = sum(f == 0.5) - s['pf2_ok'][learning_k]
        s['pf3_ok'][learning_k] = sum(np.logical_and(f == 0.7, selection))
        s['pf3_fail'][learning_k] = sum(f == 0.7) - s['pf3_ok'][learning_k]

        s['cr1_ok'][learning_k] = sum(np.logical_and(cr == 0.1, selection))
        s['cr1_fail'][learning_k] = sum(cr == 0.1) - s['cr1_ok'][learning_k]
        s['cr2_ok'][learning_k] = sum(np.logical_and(cr == 0.5, selection))
        s['cr2_fail'][learning_k] = sum(cr == 0.5) - s['cr2_ok'][learning_k]
        s['cr3_ok'][learning_k] = sum(np.logical_and(cr == 0.9, selection))
        s['cr3_fail'][learning_k] = sum(cr == 0.9) - s['cr3_ok'][learning_k]

        selection = np.array(selection)
        s['rand_ok'][learning_k] = sum(selection[rand_mut_idx])
        s['rand_fail'][learning_k] = len(rand_mut_idx) - s['rand_ok'][learning_k]
        s['best_ok'][learning_k] = sum(selection[best_mut_idx])
        s['best_fail'][learning_k] = len(best_mut_idx) - s['best_ok'][learning_k]

        s['bin_ok'][learning_k] = sum(selection[bin_cross_idx])
        s['bin_fail'][learning_k] = len(bin_cross_idx) - s['bin_ok'][learning_k]
        s['exp_ok'][learning_k] = sum(selection[exp_cross_idx])
        s['exp_fail'][learning_k] = len(exp_cross_idx) - s['exp_ok'][learning_k]
        s['saepsde_ok'][learning_k] = len(selection)
        s['saepsde_fail'][learning_k] = population_size - len(selection)

        learning_k = (learning_k + 1) % learning_period

        current_generation += 1

        if current_generation > learning_period:
            sf1 = np.sum(s['pf1_ok']) / (np.sum(s['pf1_ok']) + np.sum(s['pf1_fail'])) + 0.02
            sf2 = np.sum(s['pf2_ok']) / (np.sum(s['pf2_ok']) + np.sum(s['pf2_fail'])) + 0.02
            sf3 = np.sum(s['pf3_ok']) / (np.sum(s['pf3_ok']) + np.sum(s['pf3_fail'])) + 0.02

            pf1 = sf1 / (sf1 + sf2 + sf3)
            pf2 = sf2 / (sf1 + sf2 + sf3)
            pf3 = sf3 / (sf1 + sf2 + sf3)

            pcr1 = np.sum(s['cr1_ok']) / (np.sum(s['cr1_ok']) + np.sum(s['cr1_fail'])) + 0.02
            pcr2 = np.sum(s['cr2_ok']) / (np.sum(s['cr2_ok']) + np.sum(s['cr2_fail'])) + 0.02
            pcr3 = np.sum(s['cr3_ok']) / (np.sum(s['cr3_ok']) + np.sum(s['cr3_fail'])) + 0.02

            cr1 = pcr1 / (pcr1 + pcr2 + pcr3)
            cr2 = pcr2 / (pcr1 + pcr2 + pcr3)
            cr3 = pcr3 / (pcr1 + pcr2 + pcr3)

            pmut_best = np.sum(s['best_ok']) / (np.sum(s['best_ok']) + np.sum(s['best_fail'])) + 0.02
            pmut_rand = np.sum(s['rand_ok']) / (np.sum(s['rand_ok']) + np.sum(s['rand_fail'])) + 0.02

            best_mutation = pmut_best / (pmut_best + pmut_rand)

            pcross_bin = np.sum(s['bin_ok']) / (np.sum(s['bin_ok']) + np.sum(s['bin_fail'])) + 0.02
            pcross_exp = np.sum(s['exp_ok']) / (np.sum(s['exp_ok']) + np.sum(s['exp_fail'])) + 0.02

            bin_cross = pcross_bin / (pcross_bin + pcross_exp)

            p_saepsde = np.sum(s['saepsde_ok']) / (np.sum(s['saepsde_ok']) + np.sum(s['saepsde_fail'])) + 0.02
            p_mmts = np.sum(s['mmts_ok']) / (np.sum(s['mmts_ok']) + np.sum(s['mmts_fail'])) + 0.02

            p_mmts = p_mmts / (p_saepsde + p_mmts)

        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best]
