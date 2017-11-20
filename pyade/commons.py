import numpy as np


def keep_bounds(population: np.array, bounds: np.array) -> np.array:
    """
    Constrains the population to its proper limits.
    Any value outside its bounded ranged is clipped.
    :param population: Current population that may not be constrained.
    :type population: np.array
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.array
    :rtype np.array
    :return: Population constrained within its bounds.
    """
    minimum = [bound[0] for bound in bounds]
    maximum = [bound[1] for bound in bounds]
    return np.clip(population, minimum, maximum)


def init_population(population_size: int, individual_size: int,
                     bounds: np.array) -> np.array:
    """
    Creates a random population within its constrained bounds
    :param population_size: Number of individuals desired in the population.
    :type population_size: int
    :param individual_size: Number of features/gens
    :type individual_size: int
    :param bounds: Numpy array of tuples (min, max).
                   Each tuple represents a gen of an individual.
    :type bounds: np.array
    :rtype: np.array
    :return: Initialized population.
    """
    population = np.random.randn(population_size, individual_size)
    return keep_bounds(population, bounds)
