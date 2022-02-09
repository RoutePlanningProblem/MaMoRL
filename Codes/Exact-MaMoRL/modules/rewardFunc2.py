import numpy as np
import math

def reward(SoA, shipNewPos, n, actionOut, speedOut, notExploredDir, goalDir, outDeg) :
    retVal = 0.0
    for ship in range(0, n):
            retVal += ((0.2525 * ((speedOut[ship][ship] + 10) ** 2) -  0.16307 * (speedOut[ship][ship] + 10))) # fuel equation
    retVal = retVal /  (((0.2525 * ((10 + 10) ** 2) -  0.16307 * (10 + 10))) * n)
    return retVal

