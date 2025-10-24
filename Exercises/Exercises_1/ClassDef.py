import math
import random as rnd
import numpy as np

class Boson():

    isFermion = False #class level attribute
    def __init__(self, name, spin, momentum):
        #instance level attributes
        self.name = name
        self.momentum = momentum
        self.spin = spin
    def PrintInfo(self):
        print(f"name:{self.name},spin:{self.spin},momentum:{self.spin}")

class Higgs(Boson):

    MassSigma = 1 #class level attribute for Higgs class
    #Higgs inherits from the Boson
    def __init__(self, name, spin, momentum, MassMean):
        Boson.__init__(self,name,spin, momentum)
        #additional instance level attributes for Higgs class
        self.MassMean = MassMean

    def Energy(self):
        mass = np.random.normal(self.MassMean, self.MassSigma)
        return math.sqrt(pow(self.momentum,2)+pow(mass,2))
