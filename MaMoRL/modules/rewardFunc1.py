from numba import typed, typeof, int32, njit, float32, jit, vectorize, prange
import math
import numpy as np

# @njit
# def reward(SoA, shipNewPos, n, actionOut, speedOut, notExploredDir, goalDir) :
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

@njit
def reward(SoA, shipNewPos, n, actionOut, speedOut, notExploredDir, goalDir):
    retVal = 0.0
    crashed = np.zeros(n)
    for i in range(0, n):
        if actionOut[i][i] == goalDir[i]:
            retVal += 6 * (n / 2.4)
        retVal += 1 * SoA[i]
        if notExploredDir[i][actionOut[i][i]]:
            retVal += 2 / (6)

        for j in range(i, n):
            if i != j:
                if shipNewPos[i] == shipNewPos[j]:
                    # retVal += 1 / (48 * n)
                    retVal -= SoA[i]
                    crashed[i] = 1
        if not crashed[i]:
            retVal += 2 / 4

    retVal = retVal / (6 * n)
    return retVal, crashed