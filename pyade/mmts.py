import cec2014
import numpy as np
import pyade.commons
from typing import Callable, List, Union


def local_search_1(individual: np.ndarray, reset_sr: np.ndarray, search_range: Union[int, float], improve: np.ndarray,
                   k: int, func: Callable, fitness: float, best_solution, best_fitness):
    grade = 0

    num_evals = 0
    if not improve[k]:
        search_range /= 2
        if search_range < 1e-5:
            search_range = reset_sr
    improve[k] = False
    new_individual = individual.copy()
    current_fitness = fitness

    for i in range(len(individual)):
        new_individual[i] -= search_range
        new_fitness = func(new_individual)
        num_evals += 1
        if new_fitness < best_fitness:
            grade += 10
            best_fitness = new_fitness
            current_fitness = new_fitness
            best_solution = new_individual.copy()

        if new_fitness == current_fitness:
            new_individual = individual.copy()
            current_fitness = fitness
        else:
            if new_fitness > current_fitness:
                new_individual = individual.copy()
                current_fitness = fitness
                new_individual[i] += 0.5 * search_range
                new_fitness = func(new_individual)
                num_evals += 1

                if new_fitness < best_fitness:
                    grade += 1
                    best_fitness = new_fitness
                    best_solution = new_individual.copy()
                if new_fitness >= current_fitness:
                    current_fitness = fitness
                    new_individual = individual.copy()
                else:
                    current_fitness = new_fitness
                    grade += 1
                    improve[k] = True
            else:
                current_fitness = new_fitness
                grade += 1
                improve[k] = True

    return grade, best_solution, best_fitness, new_individual, current_fitness, num_evals

def local_search_2(individual: np.ndarray, reset_sr: np.ndarray, search_range: Union[int, float], improve: np.ndarray,
                   k: int, func: Callable, fitness: float, best_solution, best_fitness):
    grade = 0
    num_evals = 0
    if not improve[k]:
        search_range /= 2
        if search_range < 1e-5:
            search_range = reset_sr
    improve[k] = False
    new_individual = individual.copy()
    current_fitness = fitness
    for l in range(len(individual)):
        r = np.random.choice(np.array([0, 1, 2, 3]), len(individual))
        d = np.random.choice(np.array([-1, 1]), len(individual))
        for i in range(len(individual)):
            if r[i] == 0:
                new_individual[i] -= search_range * d[i]

        new_fitness = func(new_individual)
        num_evals += 1
        if new_fitness < best_fitness:
            grade += 10
            best_fitness = new_fitness
            best_solution = new_individual.copy()

        if new_fitness == current_fitness:
            new_individual = individual.copy()
        else:
            if new_fitness > current_fitness:
                new_individual = individual.copy()
                current_fitness = fitness
                for i in range(len(individual)):
                    if r[i] == 0:
                        new_individual[i] += search_range * d[i] * .5
                new_fitness = func(new_individual)
                num_evals += 1
                if new_fitness < best_fitness:
                    grade += 10
                    best_fitness = new_fitness
                    best_solution = new_individual.copy()

                if new_fitness >= current_fitness:
                    current_fitness = fitness
                    new_individual = individual.copy()
                else:
                    current_fitness = new_fitness
                    grade += 1
                    improve[k] = True
            else:
                current_fitness = new_fitness
                grade += 1
                improve[k] = True

    return grade, best_solution, best_fitness, new_individual, current_fitness, num_evals


def local_search_3(individual: np.ndarray, reset_sr: np.ndarray, search_range: Union[int, float], improve: np.ndarray,
                   k: int, func: Callable, fitness: float, best_solution, best_fitness):
    grade = 0
    num_evals = 0
    current_fitness = fitness

    x = individual.copy()
    y = individual.copy()
    z = individual.copy()
    new_individual = individual.copy()

    for i in range(len(individual)):
        x[i] += 0.1
        y[i] -= 0.1
        z[i] += 0.2
        x_fitness = func(x)
        y_fitness = func(x)
        z_fitness = func(x)
        num_evals += 3

        if x_fitness < best_fitness:
            grade += 10
            best_fitness = x_fitness
            best_solution = x.copy()

        if y_fitness < best_fitness:
            grade += 10
            best_fitness = y_fitness
            best_solution = y.copy()

        if z_fitness < best_fitness:
            grade += 10
            best_fitness = z_fitness
            best_solution = z.copy()

        d_x = fitness - x_fitness
        d_y = fitness - y_fitness
        d_z = fitness - z_fitness

        if d_x > 0:
            grade += 1

        if d_y > 0:
            grade += 1

        if d_z > 0:
            grade += 1

        a = np.random.choice(np.array([0.4, 0.5]))
        b = np.random.choice(np.array([0.1, 0.3]))
        c = np.random.choice(np.array([0, 1]))

        new_individual[i] += a * (d_x - d_y) + b * (d_z - 2 * d_x) + c
        new_fitness = func(new_individual)
        num_evals += 1

        if new_fitness >= current_fitness:
            current_fitness = fitness
            new_individual = individual.copy()
        else:
            grade += 1
            current_fitness = new_fitness

        return grade, best_solution, best_fitness, new_individual, func(new_individual), num_evals

def mmts(population: np.ndarray, bounds: np.ndarray, fitness: np.ndarray, max_evals: int, func):
    enable = np.ones(population.shape[0], np.bool)
    improve = np.ones(population.shape[0], np.bool)
    minimum, maximum = bounds[0]
    search_range = (maximum - minimum) / 2
    reset_sr = np.ones(population.shape[0]) * .4 * search_range
    grades = np.zeros(population.shape[0])
    ls1_grades = np.zeros(population.shape[0])
    ls2_grades = np.zeros(population.shape[0])
    ls3_grades = np.zeros(population.shape[0])

    best_solution = population[0].copy()
    best_fitness = fitness[np.argmin(fitness)]

    num_evals = 0
    num_test = 3
    num_best = 3
    num_ls = 5
    num_foreground = 5

    while num_evals <= max_evals:
        for i in range(population.shape[0]):
            if enable[i]:
                grades[i] = 0
                ls1_grades[i] = 0
                ls2_grades[i] = 0
                ls3_grades[i] = 0
                for j in range(num_test):
                    ls1 = local_search_1(population[i], reset_sr[i], search_range, improve, i, func,
                                         fitness[i], best_solution, best_fitness)
                    ls1_grades[i] += ls1[0]
                    best_solution = ls1[1]
                    best_fitness = ls1[2]
                    population[i] = ls1[3]
                    fitness[i] = ls1[4]
                    num_evals += ls1[5]

                    ls2 = local_search_2(population[i], reset_sr[i], search_range, improve, i, func,
                                         fitness[i], best_solution, best_fitness)
                    ls2_grades[i] += ls2[0]
                    best_solution = ls2[1]
                    best_fitness = ls2[2]
                    population[i] = ls2[3]
                    fitness[i] = ls2[4]
                    num_evals += ls2[5]

                    ls3 = local_search_3(population[i], reset_sr[i], search_range, improve, i, func,
                                         fitness[i], best_solution, best_fitness)
                    ls3_grades[i] += ls3[0]
                    best_solution = ls3[1]
                    best_fitness = ls3[2]
                    population[i] = ls3[3]
                    fitness[i] = ls3[4]
                    num_evals += ls3[5]

                my_max = max(ls1_grades[i], ls2_grades[i], ls3_grades[i])

                for j in range(num_ls):
                    if my_max == ls1_grades[i]:
                        search_k = local_search_1(population[i], reset_sr[i], search_range, improve, i, func,
                                                 fitness[i], best_solution, best_fitness)
                    elif my_max == ls2_grades[i]:
                        search_k = local_search_2(population[i], reset_sr[i], search_range, improve, i, func,
                                       fitness[i], best_solution, best_fitness)
                    else:
                        search_k = local_search_3(population[i], reset_sr[i], search_range, improve, i, func,
                                                  fitness[i], best_solution, best_fitness)

                    grades[i] += search_k[0]
                    best_solution = search_k[1]
                    best_fitness = search_k[2]
                    population[i] = search_k[3]
                    fitness[i] = search_k[4]
                    num_evals += search_k[5]

        for j in range(num_best):
            search = local_search_1(best_solution, reset_sr[i], search_range, improve, i, func,
                                             best_fitness, best_solution, best_fitness)
            best_solution = search[1]
            best_fitness = search[2]
            num_evals += search[5]

        enable = np.zeros(population.shape[0], np.bool)
        best_grades = np.argsort(grades)[::-1][:num_foreground]
        enable[best_grades] = True

        population[best_grades[-1]] = best_solution
        fitness[best_grades[-1]] = best_fitness

    return population, fitness, num_evals


if __name__ == '__main__':
    np.random.seed(0)
    bounds = np.array([[-100, 100]] * 30)
    my_population = pyade.commons.init_population(5, 30, np.array([[-100, 100]] * 30))
    bench = cec2014.Benchmark(1)
    func = lambda x: bench.get_fitness(x)
    my_fitness = pyade.commons.apply_fitness(my_population, func)
    print(my_fitness)
    print(min(my_fitness))
    a = mmts(my_population, bounds, my_fitness, 200, func)

    print(a[3])

