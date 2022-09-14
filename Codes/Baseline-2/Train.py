import csv
import datetime
import numpy as np
from random import randint
import time
import importlib


# Change below for desired values
nShip = 2 #Put number of ships here
maxSpeed = np.array ([5]*nShip) #Set the max speed here, np array size must equal number of ships
epoch = 10 #number of epoch to train model
a = 0.001
B = np.random.randint(80, 90, nShip) / 10000
graphLen = 40  # Must keep low for better training
outDeg = 7 #Set outdegree here
fileName = "inputs/Varying_Degree/704nodes_1399edges_degree7.csv" #Set input grid file here

# Automatically filled
mod = []

def loadModules():  # Load reward Modules from modules folder
    global mod
    for rCount in range(1, 4):

        # Load functions titled rewardFunc* from modules folder
        mod.append(getattr(importlib.import_module('modules.rewardFunc' + str(rCount)), 'reward'))


def loadGraph():  # load graph from csv
    # Load data from file
    global outDeg
    with open(fileName) as csvfile:
        rawData = list(csv.reader(csvfile, delimiter=','))

    #LOAD each line of csv one by one, remove first line as its the headers containing column heading for line. Put in list.
    #Each element in list is one node's information
    rawData.pop(0)
    retLink = -np.ones((len(rawData), outDeg), dtype=np.int64)  # Links from a node
    retDist = -np.ones((len(rawData), outDeg), dtype=np.int64)  # Weights of Links
    retMax = np.full(len(rawData), outDeg-1, dtype=np.int64)  # Outdegree of a node
    rawPoints = -np.ones(len(rawData), dtype=np.int64)  # Storing the nodes in raw node ID

    n = 0

    # csv to array
    for point in rawData:

        retMax[n] = outDeg
        j = 0
        # Load points from data
        for i in range(0, outDeg):
            if (point[i * 2 + 3] == 'N/A' or point[i * 2 + 3] == ''):  # stop if 'N/A'
                break

            retLink[n][j] = int(float(point[i * 2 + 3]))
            retDist[n][j] = int(float(point[i * 2 + 3 + 1]))
            j += 1
        retMax[n] = j
        rawPoints[n] = point[0]
        n += 1


    for i in range(0, len(rawPoints)):
        for j in range(0, outDeg):
            if (retLink[i][j] == -1):
                break
            retLink[i][j] = np.where(rawPoints == retLink[i][j])[0]

    global graphLen #Set above

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



def normalTrain(graphLink, graphDist, graphMax, qTable, qErrTemp, reward, v, a, B, epoch):  # Train with action selection and P table

    global outDeg
    nQ = 0.0

    nActions = outDeg + 1 # maximum number of actions
    maxOutDegree = outDeg
    iters = 5000   # Set number of iterations per epoch here

    qValRet = np.zeros(iters * nShip)
    qErrRet = np.zeros(iters * nShip)


    for iter in range(0, iters):

        # Randomize states
        state = np.random.choice(graphLen,nShip)  # Random Sample states
        randomOutDegree = np.random.randint(0, nActions, (nShip, nActions, maxOutDegree + 1, 2))  # Randomize outdegree and store ship crash directions
        notExploredDir = np.random.randint(0, 2, (nShip, nActions - 1))  # Direction to Nearest edge node
        goalDir = np.random.randint(-32, nActions - 1,nShip)  # Direction to goal if sensed (-32 to nActions to give chance of 6 / 48)

        # Initialize for storing
        actionOut = np.full((nShip, nShip), nActions - 1)  # Store actions
        speedOut = np.full((nShip, nShip), nActions - 1)  # Store speed actions
        actionOutSensed = np.full(nShip, nActions - 1)  # Store Outdegree values for actions
        newState = np.full(nShip, 0)  # store state calculated by p table
        shipProximityQ = np.zeros(nShip)  # Store Q Proximity

        for i in range(0, nShip):
            stateOut = np.full(nShip, maxOutDegree)  # Store states with respect to ship i

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
                        qArrOut[j + 4] = getShipProximityQ(graphLink[state[i]][action], state, graphLink, graphMax, i, 1)  # proximity
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

            # New state
            newState[i], _ = actionToPos(state[i], actionOut[i][i], graphLink, graphDist, graphMax)
            actionOutSensed[i] = randomOutDegree[i, actionOut[i][i]][0][0]

        if epoch >= 9:
            speedOut = np.random.randint(0, nActions, (nShip, nShip))  # Store speed actions

        # get reward and recalculate weights
        for i in range(0, nShip):
            # get reward
            qOut = reward(actionOutSensed, newState, nShip, actionOut, speedOut, notExploredDir, goalDir,shipProximityQ,outDeg)
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

            nQ += 1


    return qTable, nQ,  qValRet, qErrRet


if __name__ == "__main__":

    # init
    loadModules()  #Loads the reward function from each reward module we provide
    graphLink, graphDist, graphMax = loadGraph() # Load graph
    qTable = initQ()  # initializes Q Table

    # Normal Train Q
    qErrT = 1
    average = 0
    time_total = 0.0

    # Start time
    ct = datetime.datetime.now()
    print("start  time:-", ct)


    for i in range(1, epoch + 1):  #Epochs are set above
        print("START", i)
        print("a:", a)
        start = int(round(time.time())) / 60
        breakVal = False
        for v in range(0, len(mod)):
            qErr = np.zeros(2, dtype=np.float64)
            qTable, nQ,  qValRet, qErrRet = normalTrain(graphLink, graphDist, graphMax, qTable, qErr, mod[v], v, a, B, i)
            qErr = qErr / nQ  # calculate error

            print(qErr)
            print(nQ)
            print("Q ERROR - vector " + str(v) + ":", qErr)

            k = 0
            if v != 0:
                k = 1

        end = int(round(time.time())) / 60
        time_total += end - start
        average = (average * (i / (i + 1))) + ((end - start) / (i + 1))
        timeRem = (epoch - i) * average

        print(i, "Time Remaining:", timeRem)


    # End time
    st = datetime.datetime.now()
    print("end  time:-", st)

    #directory to save data
    np.save("data/qData.npy", qTable)

