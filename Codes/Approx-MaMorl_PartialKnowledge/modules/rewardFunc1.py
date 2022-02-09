
import math
import numpy as np

def reward(SoA, shipNewPos, n, actionOut, speedOut, notExploredDir, goalDir, proximityStore,outDeg, seenNodes):
    retVal = 0.0
    for i in range(0, n):
        if actionOut[i][i] == outDeg:
            retVal += 1 / 24

        if proximityStore[i] == 1:
            retVal += -6 * (n / 2.5)

        if actionOut[i][i] == goalDir[i]:
            retVal += 1 / 12

        if actionOut[i][i] in seenNodes:
            retVal+=-6 * (n / 2.5)

        if actionOut[i][i] != outDeg:
            retVal += 1 * SoA[i]
            if notExploredDir[i][actionOut[i][i]]:
                retVal += 1 / 6

    retVal = retVal / (outDeg * n)
    return retVal