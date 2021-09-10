
import csv
import datetime

import numpy as np
from random import randint
import json
import time
import importlib
from matplotlib import pyplot as plt

# Change for desired values
nShip = 6
maxSpeed = np.random.randint(1, 3, nShip) * 5
epoch = 10
a = 0.001
B = np.random.randint(80, 90, nShip) / 10000

# Automatically filled
mod = []
graphLen =   80# Must keep low for better training
outDeg = 9

rawTest = [24143, 24216, 24218, 24231, 24246, 24530, 24651, 24694, 24696, 24814, 25093, 25107, 25110, 25201, 25361,
           26781,
           27088, 27154, 27256, 27311, 27505, 27576, 27611, 27678, 27848, 27870, 27917, 27920, 27977, 27992, 28025,
           28214,
           28241, 30748, 31373, 31497, 31730, 33185, 33689, 34770, 34860, 35008, 35053, 35594, 35596, 35657, 35730,
           36478,
           36506, 36520, 36807, 37118, 42179, 42262, 42294, 42427, 42813, 43209]


def loadModules():  # Load reward Modules from modules folder
    global mod
    for rCount in range(1, 4):
        # rCount = 1 # This determines which reward module is loaded

        # Load functions titled rewardFunc* from modules folder
        mod.append(getattr(importlib.import_module('modules.rewardFunc' + str(rCount)),
                           'reward'))  # 11 Add to MOD - rewardFuncX.reward function X - reward number



def loadGraph():  # load graph from csv
    # Load data from file
    global outDeg
    with open("input\\1000nodes_1750edges_degree9.csv") as csvfile:
        rawData = list(csv.reader(csvfile, delimiter=','))
    rawData.pop(
        0)  # 11 LOAD each line of csv one by one, remove first line as its the headers containing column heading for line. Put in list. Each element in list is one node's information

    retLink = -np.ones((len(rawData), outDeg),
                       dtype=np.int64)  # Links from a node    11 - no of rows - no of nodes and 6 columns. Meaning for each node, we keep track of neighbour. Fill with -1's for all for now
    retDist = -np.ones((len(rawData), outDeg),
                       dtype=np.int64)  # Weights of Links  11 - no of rows - no of nodes and 6 columns. Meaning for each node, we keep track of weight of links to neighbours. Fill with -1's for all for now
    retMax = np.full(len(rawData), outDeg-1,
                     dtype=np.int64)  # Outdegree of a node  11- array of one row, each column has a value of 5 (he updates it later in for loop)
    rawPoints = -np.ones(len(rawData),
                         dtype=np.int64)  # Storing the nodes in raw node ID  11- array of one row, each column has value of -1

    n = 0

    # csv to array

    # 11 - go through each line of csv stored as list in rawdata

    '''
    11:
    Here we do the following:

    1) Remember each line of the csv is a list and list of such lists == rawData
    2) Iterate through each such line/list 
    3) Extract out the neighbor info and put into retLink (it puts the neighbour's node id) 
    4) Extract out the distance to neighbour info for given node and put into retDist 
    5) Do this until we have no more neighbours to store 
    Retmax is also updated to reflect the number of outdegree/ neighbours to a given node and stored in array 
    rawPoints is updated with the NODE ID of each node we process and is stored in that array 

    Changing number of neighbours would be done here

    '''

    for point in rawData:
        # test
        # if int(point[0]) in rawTest:
        retMax[n] = outDeg
        j = 0
        # Load points from data
        for i in range(0, outDeg):
            if (point[i * 2 + 3] == 'N/A' or point[i * 2 + 3] == ''):  # stop if 'N/A'
                break
            # test
            # if int(point[i*2 + 3]) in rawTest:
            #print(point[i * 2 + 3])
            retLink[n][j] = int(float(point[i * 2 + 3]))
            retDist[n][j] = int(float(point[i * 2 + 3 + 1]))
            j += 1
        retMax[n] = j
        rawPoints[n] = point[0]
        n += 1
    # init links to node id 0 - n

    '''
    11:
    1) Go through each node
    2) For each node, check each neighbour node ID from retlink array
    If we get -1, means that we have finished checking each neighbour of present node and move to next 
    If not, 

    '''

    for i in range(0, len(rawPoints)):
        for j in range(0, outDeg):
            if (retLink[i][j] == -1):
                break
            retLink[i][j] = np.where(rawPoints == retLink[i][j])[0]  # QUESTION : what is veing stored here

    global graphLen
    #graphLen = len(rawTest)  #11: number of nodes of graph QUESTION: why is graphLen set to rawTest
    #graphLen = 20
    # retLink, retDist, retMax = sampleGraph(80, retLink, retDist, retMax) # Sample a smaller graph for collision
    # print(retMax)
    return retLink, retDist, retMax


# Do not use this for now
def sampleGraph(size, oldLink, oldDist, oldMax):  # Sample a smaller graph for collision
    nextNode = [randint(0, len(oldLink) - 1)]
    nextNode = [0]
    nodeList = [nextNode[0]]
    global outDeg
    while len(nodeList) < size:
        nextIterNode = []
        for i in nextNode:
            for j in range(0, oldMax[i]):
                if oldLink[i][j] not in nextIterNode:
                    nextIterNode.append(oldLink[i][j])
            if i not in nodeList:
                nodeList.append(i)
        nextNode = nextIterNode.copy()

    retLink = -np.ones((len(nodeList), outDeg), dtype=np.int64)  # Links from a node
    retDist = -np.ones((len(nodeList), outDeg), dtype=np.int64)  # Weights of Links
    retMax = np.full(len(nodeList), outDeg-1, dtype=np.int64)  # Outdegree of a node
    rawPoints = -np.ones(len(nodeList), dtype=np.int64)  # Storing the nodes in raw node ID
    n = 0

    # csv to array
    for point in nodeList:
        retMax[n] = outDeg
        j = 0
        # Load points from data
        for i in range(0, oldMax[point]):
            if oldLink[point][i] in nodeList:
                retLink[n][j] = oldLink[point][i]
                retDist[n][j] = oldDist[point][i]
                j += 1
        retMax[n] = j
        rawPoints[n] = point
        n += 1

    # init links to node id 0 - n
    for i in range(0, len(rawPoints)):
        for j in range(0, outDeg):
            if (retLink[i][j] == -1):
                break
            retLink[i][j] = np.where(rawPoints == retLink[i][j])[0]

    global graphLen
    graphLen = len(nodeList)

    return retLink, retDist, retMax


def initPTable():  # init P table
    global outDeg
    pTable = np.full((nShip, nShip, len(mod), 2, 5), 1, dtype=np.float64)

    return pTable


def initQ():  # init Q table
    global outDeg
    qTable = np.full((nShip, len(mod), 2, 6), 1, dtype=np.float64)
    return qTable



def actionToPos(pos, action, graphLink, graphDist, graphMax):  # convert action taken to state
    global outDeg
    if (action == outDeg):
        return pos, 0
    # print(pos,action)
    return graphLink[pos][action], graphDist[pos][action]



def getShipProximityP(pos, shipPos, graphLink, graphMax, maxNotSeen, ship, i, n):  # Proximity calculation for P table
    global outDeg
    # Recursively check if ship in position p at n hops from pos
    # Update maxNotSeen with the same
    for p in range(0, outDeg):
        if n == 2:
            maxNotSeen[ship, i, :, p + 1] = 0
        if p < graphMax[pos]:
            for j in range(0, nShip):
                if j != i:
                    if graphLink[pos][p] == shipPos[j]:
                        maxNotSeen[ship, i, :, p + 1] = 1
            if n != 1:
                getShipProximityP(graphLink[pos][p], shipPos, graphLink, graphMax, maxNotSeen, ship, i, n - 1)



def getShipProximityQ(pos, shipPos, graphLink, graphMax, i, n):  # Proximity calculation for Q table
    global outDeg
    # Recursively check if ship in position p at n hops from pos
    for p in range(0, outDeg):
        if p < graphMax[pos]:
            for j in range(0, nShip):
                if j != i:
                    if graphLink[pos][p] == shipPos[j]:
                        return 1
            if n != 1:
                if getShipProximityQ(graphLink[pos][p], shipPos, graphLink, graphMax, i, n - 1):
                    return 1
    return 0



def normalTrain(graphLink, graphDist, graphMax, pTable, qTable, qErrTemp, pErrTemp, reward, v, a, B,
                epoch):  # Train with action selection and P table
    global outDeg
    nQ = 0.0
    nP = 0.0
    nActions = outDeg + 1  # 11- go to any of the 6 neighbors or stay in place == 7
    maxOutDegree = outDeg   # 11 - Max as per input data. Change here if number of neighbours nodes changes in input
    iters = 5000

    qValRet = np.zeros(iters * nShip)
    qErrRet = np.zeros(iters * nShip)
    pValRet = np.zeros(iters * nShip * nShip)
    pErrRet = np.zeros(iters * nShip * nShip)

    for iter in range(0, iters):
        # Randomize states
        # print(graphLen)
        state = np.random.choice(graphLen,nShip)  # Random Sample states   # 11 - sample 9 values from graph size ===> I think this denotes we sample out a set of initial nodes for each ship in given iteration of training


        #lst = [446, 20, 59, 477, 440, 346, 315, 447, 399, 226, 224, 304, 69, 101, 130, 258, 269, 457, 281, 475, 117, 236, 422, 6, 424, 303, 16, 166, 293, 247, 439, 186, 288, 368, 24, 37, 483, 107, 341, 94, 83, 65, 230, 275, 49, 474, 167, 217, 429, 276, 375, 204]
        #state = np.random.choice(lst,nShip,replace=False)

        #print(state)
        randomOutDegree = np.random.randint(0, nActions, (
        nShip, nActions, maxOutDegree + 1, 2))  # Randomize outdegree and store ship crash directions
        notExploredDir = np.random.randint(0, 2, (nShip, nActions - 1))  # Direction to Nearest edge node
        goalDir = np.random.randint(-32, nActions - 1,
                                    nShip)  # Direction to goal if sensed (-32 to nActions to give chance of 6 / 48)
        # goalDir = np.random.randint(-3, nActions - 1, nShip)
        # Initialize for storing
        actionOut = np.full((nShip, nShip), nActions - 1)  # Store actions
        speedOut = np.full((nShip, nShip), nActions - 1)  # Store speed actions
        actionOutSensed = np.full(nShip, nActions - 1)  # Store Outdegree values for actions
        newState = np.full(nShip, 0)  # store state calculated by p table
        # shipProximityP = np.random.randint(0, nActions - 1, (nShip, nShip, maxOutDegree + 1, np.max(maxSpeed) + 1)) # store ship proximity values for P Array
        shipProximityP = np.random.randint(0, nActions - 1, (
        nShip, nShip, maxOutDegree + 1, outDeg + 2))  # store ship proximity values for P Array

        shipProximityQ = np.zeros(nShip)  # Store Q Proximity
        # print(np.max(maxSpeed) + 1)
        for i in range(0, nShip):
            stateOut = np.full(nShip, maxOutDegree)  # Store states with respect to ship I
            for p in range(1, maxOutDegree + 1):
                randomOutDegree[i, :, p, :] = 0
            for j in range(0, nShip):
                if (i != j):
                    maxVal = -1000
                    # get Proximity values
                    getShipProximityP(state[j], state, graphLink, graphMax, shipProximityP, i, j, 1)
                    pVal = -1000
                    # Action with largest pValue for ship j
                    for action in range(0, graphMax[state[j]] + 1):
                        # print(action)
                        if action == graphMax[state[j]]:
                            action = outDeg
                        pTemp = 0.0
                        pArrOut = np.zeros(5)  # input of feature array
                        if action == outDeg:
                            pArrOut[0] = 0
                            pArrOut[1] = 0
                            pArrOut[2] = 0
                        else:
                            # outdegree, proximity to another ship, closest edge, speed, goal direction
                            pArrOut[0] = shipProximityP[i][j][action][0]
                            pArrOut[1] = shipProximityP[i][j][action][action + 1]
                            pArrOut[2] = notExploredDir[j][action]
                        pArrOut[3] = 0
                        if goalDir[j] == action:
                            pArrOut[4] = 1
                        else:
                            pArrOut[4] = 0

                        pTemp = np.sum(pArrOut * pTable[i][j][v][0])
                        if pTemp > pVal:
                            pVal = pTemp
                            actionOut[i][j] = action

                    stateOut[j], _ = actionToPos(state[j], actionOut[i][j], graphLink, graphDist, graphMax)
                    pVal = -1000

                    # get speed value for ship j
                    if actionOut[i][j] != outDeg:
                        for speed in range(1, maxSpeed[j] + 1):
                            pTemp = 0.0
                            pArrOut = np.zeros(5)
                            pArrOut[0] = shipProximityP[i][j][actionOut[i][j]][0]
                            pArrOut[1] = shipProximityP[i][j][actionOut[i][j]][actionOut[i][j] + 1]
                            pArrOut[2] = notExploredDir[j][actionOut[i][j]]
                            pArrOut[3] = speed
                            if goalDir[j] == actionOut[i][j]:
                                pArrOut[4] = 1
                            else:
                                pArrOut[4] = 0

                            pTemp = np.sum(pArrOut * pTable[i][j][v][1])
                            if pTemp > pVal:
                                pVal = pTemp
                                speedOut[i][j] = speed
                    else:
                        speedOut[i][j] = 0

                    # update if a ship exists in direction of movement
                    if np.any(graphLink[state[i]] == stateOut[j]):
                        index = np.where(graphLink[state[i]] == stateOut[j])[0][0]
                        randomOutDegree[i, :, index, 0] = 1
                        randomOutDegree[i, :, index, 1] = speedOut[i][j]

            qVal = -1000
            # get Maximum Q value
            for action in range(0, graphMax[state[i]] + 1):
                if action == graphMax[state[i]]:
                    action = outDeg
                if action == outDeg or randomOutDegree[i][action][action + 1][0] != 0:
                    qArrOut = np.zeros(6)  # input features for q table
                    j = 1
                    if action == outDeg:
                        qArrOut[0] = 0
                        qArrOut[j] = 0
                        qArrOut[j + 1] = 0
                        qArrOut[j + 3] = 1
                    else:
                        qArrOut[0] = randomOutDegree[i][action][0][0]  # Degree
                        qArrOut[j] = notExploredDir[i][action]  # Unsensed
                        qArrOut[j + 1] = 0  # Speed
                        qArrOut[j + 3] = 0
                        qArrOut[j + 4] = getShipProximityQ(graphLink[state[i]][action], state, graphLink, graphMax, i,
                                                           1)  # proximity
                    if goalDir[i] == action:
                        qArrOut[j + 2] = 1  # Goal
                    else:
                        qArrOut[j + 2] = 0

                    qTemp = np.sum(qTable[i][v][0] * qArrOut)
                    if qTemp > qVal:
                        qVal = qTemp
                        actionOut[i][i] = action
                        if action != outDeg:
                            shipProximityQ[i] = qArrOut[j + 4]

            qVal = -1000
            # speed of ship i
            if actionOut[i][i] != outDeg:
                for speed in range(0, maxSpeed[i] + 1):
                    qArrOut = np.zeros(6)

                    j = 1
                    qArrOut[j + 0] = notExploredDir[i][actionOut[i][i]]
                    qArrOut[j + 1] = speed
                    if goalDir[i] == actionOut[i][i]:
                        qArrOut[j + 2] = 1
                    else:
                        qArrOut[j + 2] = 0

                    qTemp = np.sum(qTable[i][v][1] * qArrOut)
                    if qTemp > qVal:
                        qVal = qTemp
                        speedOut[i][i] = speed
            else:
                speedOut[i][i] = 0

            # print(actionOut)
            newState[i], _ = actionToPos(state[i], actionOut[i][i], graphLink, graphDist, graphMax)
            actionOutSensed[i] = randomOutDegree[i, actionOut[i][i]][0][0]

        if epoch >= 9:
            speedOut = np.random.randint(0, nActions, (nShip, nShip))  # Store speed actions
        # get reward and recalculate weights
        for i in range(0, nShip):
            # get reward
            qOut = reward(actionOutSensed, newState, nShip, actionOut, speedOut, notExploredDir, goalDir,
                          shipProximityQ,outDeg)
            qValRet[iter * nShip + i] = qOut
            qArrOut = np.zeros(6)

            j = 1
            qArrOut[j + 1] = 0
            if actionOut[i][i] != outDeg:
                qArrOut[j + 0] = notExploredDir[i][actionOut[i][i]]
                qArrOut[j + 4] = shipProximityQ[i]
            else:
                qArrOut[j + 0] = 0
                qArrOut[j + 3] = 1
            if goalDir[i] == actionOut[i][i]:
                qArrOut[j + 2] = 1
            else:
                qArrOut[j + 2] = 0

            Err = 0.0
            Err = qOut - np.sum((qTable[i][v][0] * qArrOut))
            qErrTemp[0] += Err ** 2  # Error of vector 0 - action reward
            qTable[i][v][0] = qTable[i][v][0] + a * (qArrOut * Err)

            qArrOut[j + 1] = speedOut[i][i]
            qErrRet[iter * nShip + i] = np.sum((qTable[i][v][1] * qArrOut))
            Err = qOut - np.sum((qTable[i][v][1] * qArrOut))
            qErrTemp[1] += Err ** 2  # Error of vector 0 - speed reward
            qTable[i][v][1] = qTable[i][v][1] + a * (qArrOut * Err)

            # Calculate teammate model error
            for j in range(0, nShip):
                if i != j:
                    pOut = 0
                    if actionOut[i][j] == actionOut[j][j]:
                        pOut = 1
                    pSpeedOut = 0
                    if speedOut[i][j] == speedOut[j][j]:
                        pSpeedOut = 1
                    Err = 0.0
                    if v == 0:
                        pValRet[iter * nShip * nShip + i * nShip + j] = pOut

                    else:
                        pValRet[iter * nShip * nShip + i * nShip + j] = pSpeedOut
                    pArrOut = np.zeros(5)
                    pArrOut[0] = shipProximityP[i][j][actionOut[i][j]][0]
                    pArrOut[1] = shipProximityP[i][j][actionOut[i][j]][actionOut[i][j] + 1]
                    if actionOut[i][j] != outDeg:
                        pArrOut[2] = notExploredDir[j][actionOut[i][j]]
                    else:
                        pArrOut[2] = 0
                    pArrOut[3] = 0
                    if goalDir[j] == actionOut[i][j]:
                        pArrOut[4] = 1
                    else:
                        pArrOut[4] = 0

                    #print("pOut: " + str(pOut))
                    #print("np.sum((pTable[i][j][v][0] * pArrOut)): " + str(np.sum((pTable[i][j][v][0] * pArrOut))))
                    #print("pArrOut: "+ str(pArrOut))
                    Err = pOut - np.sum((pTable[i][j][v][0] * pArrOut))
                    #print(Err)

                    if v == 0:
                        pErrRet[iter * nShip * nShip + i * nShip + j] = np.sum((pTable[i][j][v][0] * pArrOut))

                    if actionOut[i][j] != outDeg:
                        pErrTemp[0] += Err ** 2  # error for action
                        #print(pErrTemp[0])
                        nP += 1
                    pTable[i][j][v][0] = pTable[i][j][v][0] + B[j] * (pArrOut * Err)

                    pArrOut[3] = speedOut[i][j]

                    if v != 0:
                        pErrRet[iter * nShip * nShip + i * nShip + j] = np.sum((pTable[i][j][v][1] * pArrOut))
                    Err = pSpeedOut - np.sum((pTable[i][j][v][1] * pArrOut))
                    pTable[i][j][v][1] = pTable[i][j][v][1] + B[j] * (pArrOut * Err)
                    if actionOut[i][j] != outDeg:
                        pErrTemp[1] += Err ** 2  # error for speed
                        nP += 1
            nQ += 1


    return pTable, qTable, nQ, nP, qValRet, qErrRet, pValRet, pErrRet


if __name__ == "__main__":
    # Init
    loadModules()  # 11 Loads the reward function from each reward module we provide
    graphLink, graphDist, graphMax = loadGraph()  # 11 - prepares array with neighbour node ids, neighbour node distances, and outdegrees for each node processed from CSV . See function for details
    pTable = initPTable()  # 11- P Table initialized : QUESTION : what is layout exactly
    qTable = initQ()  # initializes Q Table # 11- Q Table initialized : QUESTION : what is layout exactly

    # Normal Train Q and P tables
    qErrT = 1
    average = 0
    time_total = 0.0

    ct = datetime.datetime.now()
    print("start  time:-", ct)


    for i in range(1, epoch + 1):  # 11 - Epoch i think is the number of episodes of training
        print("START", i)
        print("a:", a)
        start = int(round(time.time())) / 60
        breakVal = False
        for v in range(0, len(mod)):
            qErr = np.zeros(2, dtype=np.float64)
            pErr = np.zeros(2, dtype=np.float64)
            pTable, qTable, nQ, nP, qValRet, qErrRet, pValRet, pErrRet = normalTrain(graphLink, graphDist, graphMax,
                                                                                     pTable, qTable, qErr, pErr, mod[v],
                                                                                     v, a, B, i)
            qErr = qErr / nQ  # calculate error
            pErr = pErr / nP
            print(qErr,pErr)
            print(nQ,nP)
            print("Q ERROR - vector " + str(v) + ":", qErr)
            print("P ERROR - vector " + str(v) + ":", pErr)
            k = 0
            if v != 0:
                k = 1
            # Draw graphs
            '''
            if i == epoch - 1:
                plt.title("Q Table")
                plt.subplot(2, 1, 1)
                plt.ylabel("Actual Value")
                plt.xlabel("States")
                plt.plot(qValRet)
                plt.subplot(2, 1, 2)
                plt.ylabel("Predicted Value")
                plt.xlabel("States")
                plt.plot(qErrRet)
                plt.show()

                plt.title("P Table")
                plt.subplot(2, 1, 1)
                plt.ylabel("Actual Value")
                plt.xlabel("States")
                plt.plot(pValRet)
                plt.subplot(2, 1, 2)
                plt.ylabel("Predicted Value")
                plt.xlabel("States")
                plt.plot(pErrRet)
                plt.show()
        '''
        end = int(round(time.time())) / 60
        time_total += end - start
        average = (average * (i / (i + 1))) + ((end - start) / (i + 1))
        timeRem = (epoch - i) * average

        print(i, "Time Remaining:", timeRem)

    #print(qTable[:, 0, 0])  # print Q table weights for reward 1

    st = datetime.datetime.now()
    print("end  time:-", st)

    #Baseline solely uses q table
    np.save("data\qData.npy", qTable)


