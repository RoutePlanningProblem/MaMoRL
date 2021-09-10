from numba import typed, typeof, int32, njit, float32, jit, vectorize, prange
import math


@njit
def reward(SoA, shipNewPos, n, actionOut, speedOut, notExploredDir, goalDir) :
    retVal = 0.0
    for ship in range(0, n):
        if speedOut[ship][ship] != 0:
            retVal += 1 / speedOut[ship][ship]  # time equation
    retVal /= n
    return retVal,0

# @njit
# def reward(SoA, shipNewPos, n, actionOut, speedOut, notExploredDir) :
#     retVal = 0.0
#     for i in range(0, n):
#         for j in range(i, n):
#             if i != j:
#                 if shipNewPos[i] == shipNewPos[j]:
#                     retVal -= 1
#         retVal += SoA[i]
#         if notExploredDir[i][actionOut[i][i]]:
#             retVal += 1 / (7*n)
#     retVal = retVal / (6*n)
#     return retVal
