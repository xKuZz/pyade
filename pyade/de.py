import numpy as np
import pyade.commons

class DE_Settings:
    def __init__(self, func, bounds, population_size, individual_size,
                 CR = 0.3, strategy ='random', F = None, n_parents = 3,
                 max_iter = 10000, seed = None):
        self.func = func # Function for fitness
        self.bounds = bounds # Bounds for each feature of the poblation
        self.population_size = population_size # Size of the population
        self.individual_size = individual_size # Size of the individual
        self.CR = CR # Recombination chance in [0.0,1.0]
        self.strategy = strategy # Strategy selected for mutation
        self.F = F # Mutation parameter in [0,2]
        self.n_parents = n_parents # Number of parents for mutation
        self.max_iter = max_iter # Termination criteria: Maximum number of genereations
        self.seed = seed # Seed for random generation

def de(settings):
    # Extract from namedtuple
    func = settings.func
    bounds = settings.bounds
    population_size = settings.population_size
    individual_size = settings.individual_size
    CR = settings.CR
    strategy = settings.strategy
    F = settings.n_parents
    n_parents = settings.n_parents
    max_iter = settings.max_iter
    seed = settings.seed
    def __recombinate(CR, population, mutated):
        return np.array([mutated[i] if np.random.ranf() < CR else population[i] for i in range(population.shape[0])])

    def __selection(recombinated, r_fitness, population, fitness):
        return np.array([recombinated[i] if r_fitness[i] < fitness[i] else population[i] for i in range(population.shape[0])])

    # 1. Initialization
    population = pyade.commons.init_population(population_size, individual_size, bounds)
    fitness = pyade.commons.apply_fitness(population, func)

    for num_iters in range(max_iter):
        mutated = pyade.commons.binary_mutation(population, F, bounds)
        recombinated = __recombinate(CR, population, mutated)
        r_fitness = pyade.commons.apply_fitness(recombinated, func)
        population = __selection(recombinated,r_fitness,population,fitness)

    fitness = [func(individual) for individual in population]
    best = np.argmin(fitness)
    return population[best]

if __name__ == '__main__':
    f = lambda x : (x**2).sum()
    settings = DE_Settings(func=f, bounds=[(-3,3),(-5,5)],population_size=50, individual_size = 2,seed = 10)
    print(de(settings))

