import numpy as np
import pyade.commons

class DE_Settings:
    def __init__(self, func, bounds, population_size, CR = 0.3, strategy ='random', F = None, n_parents = 3,
                 max_iter = 10000, seed = None):
        self.func = func # Function for fitness
        self.bounds = bounds # Bounds for each feature of the poblation
        self.population_size = population_size # Size of the population
        self.CR = CR # Recombination chance in [0.0,1.0]
        self.strategy = strategy # Strategy selected for mutation
        self.F = F # Mutation parameter in [0,2]
        self.n_parents = n_parents # Number of parents for mutation
        self.max_iter = max_iter # Termination criteria: Maximum number of genereations
        self.seed = seed # Seed for random generation

def de(**settings):
    # Extract from namedtuple
    func = settings.func
    bounds = settings.bounds
    population_size = settings.population_size
    individual_size = settings.individual_size
    CR = settings.CR
    strategy = settings.strategy
    F = settings.n_parents
    n_parents = settings.n_parents
    max_iter = settings.max_ite
    seed = settings.seed
    def __mutate1(strategy, current_index, population, F, n_parents ):
        # Random strategy. Pull 3 candidates out of the population different from the current one
        if strategy == 'random':
            candidates = np.delete(population, current_index,0)
            candidates = candidates[np.random.randint(0,candidates.shape[0],n_parents,np.int)]

        new = candidates[2] + F * (candidates[0]-candidates[1])
        return np.array(new)

    def __mutate(strategy,population, bounds, F, n_parents = 3):
        mut = [__mutate1(strategy, individual, population, F, n_parents) for individual in range(population.shape[0])]
        mut = np.reshape(np.array(mut),population.shape)
        mut = pyade.commons._keep_bounds(mut,bounds)
        return mut

    def __recombinate(CR, population, mutated):
        return np.array([mutated[i] if np.random.ranf() < CR else population[i] for i in range(population.shape[0])])

    def __selection(recombinated, r_fitness, population, fitness):
        return np.array([recombinated[i] if r_fitness[i] < fitness[i] else population[i] for i in range(population.shape[0])])

    def __fitness(population):
        return [func(individual) for individual in population]

    # 1. Initialization
    population = pyade.commons._init_population(population_size, individual_size)

    population = np.array(population)
    # 1.4 Evaluate fitness
    fitness = __fitness(population)

    for num_iters in range(max_iter):
        mutated = __mutate(strategy,population,bounds,F,n_parents)
        recombinated = __recombinate(CR, population, mutated)
        r_fitness = [func(individual) for individual in recombinated]
        population = __selection(recombinated,r_fitness,population,fitness)

    fitness = [func(individual) for individual in population]
    best = np.argmin(fitness)
    return population[best]

if __name__ == '__main__':
    f = lambda x : (x**2).sum()
    settings = DE_Settings(func=f, bounds=[(-3,3),(-5,5)],population_size=50,seed = 10)
    print(de(settings))

