from numba import typed, typeof, int32, njit, float32, jit, vectorize, prange
import math


def reward(SoA, shipNewPos, n, actionOut, speedOut, notExploredDir, goalDir, proximityStore,outDeg):
    retVal = 0.0
    for ship in range(0, n):
        if speedOut[ship][ship] != 0:
            retVal += 1 / speedOut[ship][ship]  # time equation
    retVal /= n
    return retVal
