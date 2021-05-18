from numpy.core.fromnumeric import mean
from numpy.core.function_base import linspace
from numpy.core.numeric import ones
import simpy
import numpy.random as rnd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from simpy.core import T
from simpy.resources.resource import Request
from scipy.stats import johnsonsb

#Parameter definieren
numargs = johnsonsb.numargs 
a, b = 4.32, 3.18
rv = johnsonsb(a, b) 

print ("RV : \n", rv) 

quantile = np.arange (0.01, 1, 0.1) 

# Random Variates 
R = johnsonsb.rvs(a, b, scale = 2, size = 10) 
print ("Random Variates : \n", R)

# PDF 
R = johnsonsb.pdf(a, b, quantile, loc = 0, scale = 1) 
print ("\nProbability Distribution : \n", R)

# Representation of rnd variates
distribution = np.linspace(0, np.minimum(rv.dist.b, 3)) 
print("Distribution : \n", distribution) 

plot = plt.plot(distribution, rv.pdf(distribution)) 
plt.show()