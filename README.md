# PyADE

PyADE is a Python package that allows any user to use multiple differential evolution algorithms allowing both using them without any knowledge about what they do or to specify control parameters to obtain optimal results while your using this package.

## Library Installation
To easily install the package you can use PyPy


```bash
pip install numpy scipy pyade-python
```

## Library use
You can use any of the following algorithms: DE, SaDE, JADE, SHADE, L-SHADE, iL-SHADE, jSO, L-SHADE-cnEpSin, and MPEDE. This is an example of use of the library:
```python
# We import the algorithm (You can use from pyade import * to import all of them)
import pyade.ilshade 
import numpy as np

# You may want to use a variable so its easier to change it if we want
algorithm = pyade.ilshade 

# We get default parameters for a problem with two variables
params = algorithm.get_default_params(dim=2) 

# We define the boundaries of the variables
params['bounds'] = np.array([[-75, 75]] * 2) 

# We indicate the function we want to minimize
params['func'] = lambda x: x[0]**2 + x[1]**2 + x[0]*x[1] - 500 

# We run the algorithm and obtain the results
solution, fitness = algorithm.apply(**params)
```

Look at the library documentation to see each module name and which control parameters can be modified for each algorithm

## Optional parameters in fitness function
You can also add optional fixed parameters to the input in your fitness functions. All optional parameters must be in params['opts']. If you want to use more than just one parameter, you could use a Tuple or any other type than can handle more than one element.
By default, params['opts'] will be None, and you may use the library as in the previous example. When using params['opts'], params['func'] must take two arguments as input: the first one will be the individual to be evaluated and the second will be the optional parameter(s).

In the following example, we will set two fixed optional parameters, and change them between two executions of the algorithm.

```python
# We import the algorithm (You can use from pyade import * to import all of them)
import pyade.ilshade
import numpy as np

# You may want to use a variable so its easier to change it if we want
algorithm = pyade.ilshade

# We get default parameters for a problem with two variables
params = algorithm.get_default_params(dim=2)

# We define the boundaries of the variables
params['bounds'] = np.array([[-75, 75]] * 2)

# We indicate the function we want to minimize
params['opts'] = (2, 500)
params['func'] = lambda x, y: x[0]**2 + x[1]**y[0] + x[0]*x[1] - y[1]

# We run the algorithm and obtain the results
solution, fitness = algorithm.apply(**params)
print(fitness)

# We change the fixed optional parameters for the fitness function
params['opts'] = (2, 700)

# We run the algorithm and obtain the new results
solution, fitness = algorithm.apply(**params)
print(fitness)
```

