import numpy as np
from typing import Callable, Union


def keep_bounds(population: np.ndarray,
                bounds: Union[np.ndarray, list]) -> np.ndarray:
    """
    Constrains the population to its proper limits.
    Any value outside its bounded ranged is clipped.
    :param population: Current population that may not be constrained.
    :type population: np.ndarray
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: Union[np.ndarray, list]
    :rtype np.ndarray
    :return: Population constrained within its bounds.
    """
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    return np.clip(population, minimum, maximum)


def init_population(population_size: int, individual_size: int,
                    bounds: Union[np.ndarray, list]) -> np.ndarray:
    """
    Creates a random population within its constrained bounds.
    :param population_size: Number of individuals desired in the population.
    :type population_size: int
    :param individual_size: Number of features/gens.
    :type individual_size: int
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: Union[np.ndarray, list]
    :rtype: np.ndarray
    :return: Initialized population.
    """
    # FIXME: Adjust to real bounds
    population = np.random.randn(population_size, individual_size)
    return keep_bounds(population, bounds)


def apply_fitness(population: np.ndarray,
                  func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Applies the given fitness function to each individual of the population.
    :param population: Population to apply the current fitness function.
    :type population: np.ndarray
    :param func: Function that is used to calculate the fitness.
    :type func: np.ndarray
    :rtype np.ndarray
    :return: Numpy array of fitness for each individual.
    """
    return np.array([func(individual) for individual in population])


def binary_mutation(population: np.ndarray,
                    F: Union[int, float],
                    bounds: Union[np.ndarray, list]) -> np.ndarray:
    """
    Calculate the binary mutation of the population. For each individual (n),
    3 random parents (x,y,z) are selected. The parents are guaranteed to not
    be in the same position than the original. New individual are created by
    n = z + F * (x-y)
    :param population: Population to apply the mutation
    :type population: np.ndarray
    :param F: Parameter of control of the mutation. Must be in [0, 2].
    :type F: Union[int, float]
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: Union[np.ndarray, list]
    :rtype: np.ndarray
    :return: Mutated population
    """

    # 1. For each number, obtain 3 random integers that are not the number
    parents = []
    for i in range(population.shape[0]):
        choices = np.append(np.arange(i), np.arange(i+1, population.shape[0]))
        parents.append(np.random.choice(choices, 3, replace=False))

    parents = np.matrix(parents)

    # 2. Apply the formula to each set of parents
    mutated = F * (population[parents[:, 0]] - population[parents[:, 1]])
    mutated = population[parents[:, 2]]
    mutated = np.reshape(mutated, population.shape)
    return keep_bounds(mutated, bounds)
