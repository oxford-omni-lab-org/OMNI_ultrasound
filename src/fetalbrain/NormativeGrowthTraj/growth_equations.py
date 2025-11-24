""" Normative Growth Trajectories

Normative growth trajectories of fetal brain regions validated by satisfactory maturation of neurodevelopmental domains at 2 years of age - Nature Comms 2025

"""


import math
import numpy as np

class NormGrowthEQ:

    def __init__(self) -> None:
        pass
        """ 
        Normative Growth Trajectories for volume in cm3


        Input:  GA - the gestational age of the subject (in weeks) 
            
        
        """
    def TBV(self,GA):
        u = 1.954510 + 0.018205*GA**3 + -0.178633*GA**2
        o = np.exp(-0.702623 + 0.150265*GA)
        return u,o
    
    def CoPV(self,GA):
        u = 213.602100 + -1055185.000000*GA**-3 + 179286.700000*GA**-2 + -10465.890000*GA**-1
        o = np.exp(-3.319410 + 0.276768*GA + -0.004899*GA**2)
        return u,o
    
    def WMV(self,GA):
        u = 147.933791 + -20.171007*GA + 0.862044*GA**2 + -0.010213*GA**3
        o = np.exp(8.181534 + -34.670966*GA**-0.5)
        return u,o
    
    def DGMV(self,GA):
        u = 2.565062 + -0.905458*GA**0.5 + 0.000642*GA**3
        o = np.exp(-3.357278 + 0.134731*GA)
        return u,o
    
    def CBV(self,GA):
        u = 20.062157 + -26.210291*math.log(GA) + 13.249654*GA**0.5
        o = np.exp(-4.802530 + 0.144062*GA)
        return u,o
    
    def ChPV(self,GA):
        u = -8.747354 + 1.226594*GA + -0.052478*GA**2 + 0.000745*GA**3
        o = np.exp(-4.017505 + -1353.340000*GA**-2 + 110.794500*GA**-1)
        return u,o
    
    def LVV(self,GA):
        u = 0.199152 + -759.201300*GA**-3 + 0.000006*GA**3
        o = np.exp(-8.870769 + 1.868799*math.log(GA))
        return u,o
    
    def FHV(self,GA):
        u = 1.957187 + -0.247137*GA + 0.010655*GA**2 + -0.000135*GA**3
        o = np.exp(-28.094510 + 1841.363000*GA**-1 + 334027.300000*GA**-3 + -43795.430000*GA**-2)
        return u,o
    
    def BSV(self,GA):
        u = 27.563862 + -187.829409*GA**-0.5 + 9843.864509*GA**-2 + -77842.728897*GA**-3
        o = np.exp(41.292163 + -7.207325*math.log(GA) + -745.053409*GA**-1 + 5815.331813*GA**-2)
        return u,o
    
    def ThV(self,GA):
        u = -0.006779 + 0.000063*GA**3
        o = np.exp(2.222630 + -21.555120*GA**-0.5)
        return u,o
    
    def CSPV(self,GA):
        u = 4.356401 + -0.607443*GA + 0.026988*GA**2 + -0.000357*GA**3
        o = np.exp(-7.132420 + 0.186246*GA)
        return u,o
   

if __name__ == "__main__":

    # Set up the class containing all equations
    EQ = NormGrowthEQ()

    # Individual 
    CoP_volume = 7 #cm3
    age = 20.2 #gestational weeks

    # Compute z-score
    u,o = getattr(EQ,"CoPV")(age) # find normative mean + std
    z = (CoP_volume - u)/o 
    print("z-score for individual: ", z)
