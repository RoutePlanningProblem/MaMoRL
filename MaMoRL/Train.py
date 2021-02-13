from numba import typed, typeof, int32, njit, float32, jit, vectorize, prange
from numba.experimental import jitclass
import csv
import numpy as np
from random import randint
import math
import itertools as it
import json
import time
import importlib
from numpy.random import default_rng
from sys import getsizeof
mod = []

# load modules
for rCount in range(1, 4):

    mod.append(getattr(importlib.import_module('modules.rewardFunc' + str(rCount)), 'reward'))


displayWidth = 1600
displayHeight = 800
nShip = 2

graphLen = 710
dest = 0
testFlag = 0

graphLink = -np.ones((710, 6), dtype=np.int32)
graphDist = -np.ones((710, 6), dtype=np.int32)
graphMax = np.zeros(710, dtype=np.int32)

shipPos = np.zeros(nShip, dtype=np.int32)
maxSpeed = np.array([5, 10, 10], dtype=np.int32)
maxSpeed = np.random.randint(1, 3, nShip) * 5 #Set max speed here

statesTest = [480, 583, 677, 523, 560, 478, 378, 404, 441, 683, 692,406, 456, 449, 403, 418, 675, 537, 374, 679, 557, 576,
              377, 439, 545, 390, 702, 376, 384, 388, 577, 543, 519, 556, 455, 397, 587, 578, 446, 450, 467, 375, 469, 402,
              460, 442, 465, 452, 461, 391, 438, 531, 548, 513, 434, 521, 551, 561]

rawTest = [24143, 24216, 24218, 24231, 24246, 24530, 24651, 24694, 24696, 24814, 25093, 25107, 25110, 25201, 25361, 26781,
           27088, 27154, 27256, 27311, 27505, 27576, 27611, 27678, 27848, 27870, 27917, 27920, 27977, 27992, 28025, 28214,
           28241, 30748, 31373, 31497, 31730, 33185, 33689, 34770, 34860, 35008, 35053, 35594, 35596, 35657, 35730, 36478,
           36506, 36520, 36807, 37118, 42179, 42262, 42294, 42427, 42813, 43209]



def loadGraph(): # load graph from csv
    rawData = []
    with open('inputMap.csv') as csvfile:  #Set name of input grid here
        rawData = list(csv.reader(csvfile, delimiter=','))
    rawData.pop(0)
    retLink = -np.ones((len(rawTest), 6), dtype=np.int32)
    retDist = -np.ones((len(rawTest), 6), dtype=np.int32)
    retMax = np.full(len(rawTest), 5, dtype=np.int32)
    rawPoints = -np.ones(len(rawTest), dtype=np.int32)

    retLink = -np.ones((710, 6), dtype=np.int32)
    retDist = -np.ones((710, 6), dtype=np.int32)
    retMax = np.full(710, 5, dtype=np.int32)
    rawPoints = -np.ones(710, dtype=np.int32)
    n = 0

    #csv to array
    for point in rawData:
        # test
        # if int(point[0]) in rawTest:
        retMax[n] = 6
        j = 0
        for i in range(0, 6):
            if(point[i*2 + 3] == 'N/A'):
                break
            # test
            # if int(point[i*2 + 3]) in rawTest:
            retLink[n][j] = int(point[i*2 + 3])
            retDist[n][j] = int(point[i*2 + 3 + 1])
            j += 1
        retMax[n] = j
        rawPoints[n] = point[0]
        n += 1
    #init links
    for i in range(0, len(rawPoints)):
        for j in range(0, 6):
            if(retLink[i][j] == -1):
                break
            retLink[i][j] = np.where(rawPoints == retLink[i][j])[0]

    global graphLen
    graphLen = len(rawTest)


    return retLink, retDist, retMax


def randomLoc(): # random ship position
    retPos = np.zeros(nShip, dtype=np.int32)
    for i in range(0, nShip):
        retPos[i] = randint(0, graphLen - 1)
    return retPos


def initShip(): # init P table
    shipPos = np.zeros(nShip, dtype=np.int32)
    shipPos = randomLoc()

    pTable = np.full(((nShip, nShip) + ((graphLen, ) * nShip)+ (7, 7)), 1/7 ,dtype=np.float32)
    return shipPos, pTable

def initQ(): # init Q table
    qTable = np.full(((graphLen, ) * nShip) + ((7, 7) * nShip) , 1/7, dtype=np.float32)
    return qTable


def actionToPos(pos, action, graphLink, graphDist, graphMax): # convert action taken to state
    if(action == 6):
        return pos, 0
    return graphLink[pos][action], graphDist[pos][action]


def reward(SoA, shipNewPos, n, actionOut, notExploredDir) :
    retVal = 0.0
    for i in range(0, n):
        for j in range(i, n):
            if i != j:
                if shipNewPos[i] == shipNewPos[j]:
                    retVal -= 1
        retVal += SoA[i]
        if notExploredDir[i][actionOut[i][i]]:
            retVal += 1 / (7*n)
    retVal = retVal / (6*n)
    return retVal


def shipProximity(shipPos, graphLink, graphMax, maxNotSeen, ship, i):

    for p in range(0, 6):
        maxNotSeen[ship, i, :, p + 1] = 0
        if p < graphMax[shipPos[i]]:
            for j in range(0, nShip):
                if j != i:
                    if np.any(graphLink[p] == shipPos[j]):
                        maxNotSeen[ship, i, :, p + 1] = 1


def normalTrain(graphLink, graphDist, graphMax, pTable, qTable, qErr, pErr, reward, v, increase, state, a): # Train with action selection and P table
    B = np.random.randint(20, 30, nShip) / 100000
    Arr = [0]
    qErrTemp = np.zeros(2, dtype=np.float32)
    pErrTemp = np.zeros(2, dtype=np.float32)
    nQ = 0
    nP = 0
    contFlag = True
    state = np.array([0, 1])
    while state[0] < graphLen:
        actionOut = np.full((nShip, nShip), 6)
        speedOut = np.full((nShip, nShip), 6)
        SoA = np.full(nShip, 6)
        newState = np.full(nShip, 0)
        qArr = np.random.randint(0, 6, (nShip, 7, 7, 2))
        maxNotSeen = np.random.randint(0, 6, (nShip, nShip, 7, np.max(maxSpeed) + 1))
        notExploredDir = np.random.randint(0, 2, (nShip, 6))
        goalDir = np.random.randint(-32, 6, nShip)
        for i in range(0, nShip):
            stateOut = np.full(nShip, 6)
            for p in range(1, 7):
                qArr[i, :, p, :] = 0
            countJ = np.array([0, 0])
            for j in range(0, nShip):
                if(i != j):
                    maxVal = -1000
                    shipProximity(state, graphLink, graphMax, maxNotSeen, i, j)
                    pVal = -1000
                    for action in range(0, graphMax[state[j]] + 1):
                        if action == graphMax[state[j]]:
                            action = 6
                        pTemp = 0.0
                        pTemp = pTable[i][j][state[0]][state[1]][action][maxNotSeen[i][j][action][0]]
                        if pTemp > pVal:
                            pVal = pTemp
                            actionOut[i][j] = action
                            countJ[j] = maxNotSeen[i][j][action][0]

                    stateOut[j], _ = actionToPos(state[j], actionOut[i][j], graphLink, graphDist, graphMax)

            qVal = -1000
            actionJ = actionOut[i].copy()
            for action in range(0, graphMax[state[i]] + 1 ):
                if action == graphMax[state[i]]:
                    action = 6
                countJ[i] = qArr[i][action][0][0]
                actionJ[i] = action
                qTemp = qTable[state[0]][state[1]][actionJ[0]][countJ[0]][actionJ[1]][countJ[1]]
                if qTemp > qVal:
                    qVal = qTemp
                    actionOut[i][i] = action

            newState[i], _ = actionToPos(state[i], actionOut[i][i], graphLink, graphDist, graphMax)
            SoA[i] = qArr[i, actionOut[i][i]][0][0]

        qOut,crashed = reward(SoA, newState, nShip, actionOut, speedOut, notExploredDir, goalDir)
        qTable[state[0]][state[1]][actionOut[0][0]][qArr[0][actionOut[0][0]][0][0]][actionOut[1][1]][qArr[1][actionOut[1][1]][0][0]] *= 1 - a
        qTable[state[0]][state[1]][actionOut[0][0]][qArr[0][actionOut[0][0]][0][0]][actionOut[1][1]][qArr[1][actionOut[1][1]][0][0]] += a * qOut

        for i in range(0, nShip):
            for j in range(0, nShip):
                if i != j:
                    for action in range(0, graphMax[state[j]] + 1):
                        if action == graphMax[state[j]]:
                            action = 6
                        pTable[i][j][state[0]][state[1]][actionOut[j][i]][maxNotSeen[i][j][actionOut[j][i]][0]] += pTable[i][j][state[0]][state[1]][action][maxNotSeen[i][j][action][0]] * B[i]
                        pTable[i][j][state[0]][state[1]][action][maxNotSeen[i][j][action][0]] *= (1- B[i])

                    # For this reward

        # contFlag = True
        # for i in range(0, nShip):
        #     for j in range(i, nShip):
        #         if state[i] == state[j]:
        #             contFlag = False
        state[1] += 1
        if state[1] == graphLen:
            state[0] += 1
            state[1] = 0
        # print(state)

    return pTable, qTable, qErrTemp, pErrTemp, state



if __name__ == "__main__":
    #Init
    graphLink, graphDist, graphMax = loadGraph()
    shipPos, pTable = initShip()
    qTable = initQ()
     # Train with action selection and P table
    # iterTrain(graphLink, graphDist, graphMax, pTable, qTable,
    #             distTable)  # Train with action selection and P table

    epoch = 20
    average = 0
    state = np.zeros(nShip, dtype=np.int32)
    a = 0.001

    # Normal Train Q and P tables
    qErrT = 1
    flag = True
    rng = default_rng()
    total_time = 0.0
    for i in range(1, epoch + 1):
        print("START", i)
        print("a:", a)
        start = int(round(time.time() )) / 60
        breakVal = False
        for v in range(0, len(mod)):
            pTable, qTable, qErr, pErr, state = normalTrain(graphLink, graphDist, graphMax, pTable, qTable, 1000, 1000, mod[v], v, 1, state, a)
        #     if v == 0:
        #         if qErr[0] > qErrT:
        #             a *= 0.8
        #         else:
        #             pTable = pTableOut.copy()
        #             qTable = qTableOut.copy()
        #         qErrT = qErr[0]
        #     else:
        #         pTable = pTableOut.copy()
        #         qTable = qTableOut.copy()
        #     if v == 0 and qErr[0] <= 0.1:
        #         breakVal = True
            print("Q ERROR - vector " + str(v) + ":", qErr)
            print("P ERROR - vector " + str(v) + ":", pErr)
        # if breakVal:
        #     break

        end = int(round(time.time() )) / 60
        total_time += end - start
        average = (average * (i / (i + 1))) + ((end - start) / (i + 1))
        timeRem = (epoch - i) * average
        print(i, "Time Remaining:", timeRem)
    # print("QTABLE")
    # print(qTable)
    # print("PTABLE")
    # print(pTable)
    print(getsizeof(qTable))
    print(getsizeof(pTable))
    np.save("qData.npy", qTable)
    for i in range(0, nShip):
        for j in range(0, nShip):
            if (i != j):
                np.save(str(i) + '-' + str(j) + '-' + 'pData', pTable[i][j])

