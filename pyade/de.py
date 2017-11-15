import numpy as np
import collections

class DE_Settings:
    def __init__(self, func, bounds, population_size, GR = 0.3, strategy = 'random', F = None, n_parents = 3,
                 max_iter = 10000, seed = None):
        self.func = func # Function for fitness
        self.bounds = bounds # Bounds for each feature of the poblation
        self.population_size = population_size # Size of the population
        self.GR = GR # Recombination chance in [0.0,1.0]
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
    GR = settings.GR
    strategy = settings.strategy
    F = settings.n_parents
    n_parents = settings.n_parents
    max_iter = settings.max_iter
    seed = settings.seed
    def __mutate1(strategy, current_index, population, F, n_parents ):
        # Random strategy. Pull 3 candidates out of the population different from the current one
        if strategy == 'random':
            candidates = np.delete(population, current_index,0)
            candidates = candidates[np.random.randint(0,candidates.shape[0],n_parents,np.int)]
        if F == None:
            F = 2 * np.random.random_sample()

        new = candidates[2] + F * (candidates[0]-candidates[1])
        return np.array(new)

    def __mutate(strategy,population, bounds, F, n_parents = 3):
        mut = [__mutate1(strategy, individual, population, F, n_parents) for individual in range(population.shape[0])]
        mut = np.reshape(np.array(mut),population.shape)
        mut = __keep_bounds(mut,bounds)
        return mut

    def __keep_bounds(population, bounds):
        for i in range(len(bounds)):
            population[:,i] = np.clip(population[:,i],bounds[i][0], bounds[i][1])
        return population

    def __recombinate(GR, population, mutated):
        return np.array([mutated[i] if np.random.ranf() < GR else population[i] for i in range(population.shape[0])])

    def __selection(recombinated, r_fitness, population, fitness):
        return np.array([recombinated[i] if r_fitness[i] < fitness[i] else population[i] for i in range(population.shape[0])])

    def __fitness(population):
        return [func(individual) for individual in population]
    # 1. Initialization: Start a random population
    # 1.1 Create an array: each column represents an attribute
    #     First row is the minimum and second row is the maximum.
    limits = np.array(bounds, dtype='float').T
    scale = np.subtract(limits[1], limits[0])

    # 1.2 Set the seed for the algorithm
    np.random.seed(seed)

    # 1.3 Choose a random value for each attribute within its bounds
    population = []
    for i in range(population_size):
        factor = np.random.random_sample(len(bounds))
        individual = np.multiply(scale, factor)
        population.append(np.add(individual, limits[0]))

    population = np.array(population)
    # 1.4 Evaluate fitness
    fitness = __fitness(population)

    for num_iters in range(max_iter):
        mutated = __mutate(strategy,population,bounds,F,n_parents)
        recombinated = __recombinate(GR, population, mutated)
        r_fitness = [func(individual) for individual in recombinated]
        population = __selection(recombinated,r_fitness,population,fitness)

    fitness = [func(individual) for individual in population]
    best = np.argmin(fitness)
    return population[best]

if __name__ == '__main__':
    f = lambda x : x[0]**2+x[1]**2
    settings = DE_Settings(func=f, bounds=[(-3,3),(-5,5)],population_size=50,seed = 10)
    print(de(settings))

