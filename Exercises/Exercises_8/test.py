import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import random
from scipy.optimize import brentq

# Problem 1
def Factorial( n ):
    factorial = 1;
    if ( n > 1 ):
        for i in range(1,n+1):
            factorial *= i
    return factorial

def Binomial( r, p, N):
    bin = 0.
    bin = Factorial(N)/(Factorial(r)*Factorial(N-r))*(p**r)*(1-p)**(N-r)
    return bin

def Sum(p):
    Sum = 0
    for r in range(4):
        Sum+=Binomial(r,p,10)
    return Sum

p = np.linspace(0,1,1000)

values = Sum(p)

print(values)

p_up = p[values<0.05][0]
print(p_up)