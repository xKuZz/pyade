import numpy as np
import pyade.commons
import pyade.de
import random

random.seed(0)
np.random.seed(0)


def test_init_population():
    for i in range(10):
        pop_size = random.randint(3, 1000)
        ind_size = random.randint(3, 100)
        a = random.randint(-1000, 0)
        b = random.randint(0, 1000)
        pop = pyade.commons.init_population(pop_size, ind_size, [[a, b] * ind_size])
        assert pop.shape == (pop_size, ind_size)


def test_boundaries():
    for i in range(10):
        pop_size = random.randint(3, 1000)
        ind_size = random.randint(3, 100)
        a = random.randint(-1000, 0)
        b = random.randint(0, 1000)
        pop = pyade.commons.init_population(pop_size, ind_size, [[a, b] * ind_size])

        a /= 2
        b /= 2
        pop = pyade.commons.keep_bounds(pop, [[a, b] * ind_size])
        assert pop.min() >= a
        assert pop.min() <= b


def my_fitness(x: np.ndarray):
    return (x**2).sum()

#TODO: Comprobar que mejor, peor y media/mediana mejora al ejecutar
#TODO: Apuntar hitos para ficheros de texto

def test_differential_evolution():
    params = pyade.de.get_default_de_params()
    params['population_size'] = 50
    params['individual_size'] = 10
    params['max_iters'] = 2000
    params['f'] = 1
    params['cr'] = 0.5
    params['bounds'] = np.array([[-100, 100]] * params['individual_size'])
    params['func'] = my_fitness

    solution, fitness = pyade.de.de(**params)
    assert len(solution) == params['individual_size']
    assert fitness == params['func'](solution)