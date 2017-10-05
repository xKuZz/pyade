import numpy as np


def de(func, bounds, population_size, max_iter = 1000, seed = None):

    # 1. Initialization: Start a random population
    # 1.1 Create an array: each column represents an attribute
    #     First row is the minimum and second row is the maximum.
    limits = np.array(bounds, dtype='float').T
    scale = np.subtract(limits[1], limits[0])

    # 1.2 Set the seed for the algorithm
    np.random.seed(seed)

    # 1.3 Choose a random value for each attribute within its bounds
    factor = np.random.random_sample(population_size)
    population = np.multiply(scale, factor)
    population = np.add(population,limits[0])




de(None, [(-3,3),(-5,5)], 2, seed = 10)

