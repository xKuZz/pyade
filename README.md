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

Look at the library documentation to see each package name and which control parameters can be modified for each algorithm
