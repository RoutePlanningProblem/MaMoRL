import csv
import numpy as np
from random import randint
import time
import importlib
from numpy.random import default_rng
from sys import getsizeof
import datetime


"""
Exact MaMoRL Training Code 

"""

mod = []

# load modules containing reward functions
for rCount in range(1, 4):
    try:
        mod.append(getattr(importlib.import_module('modules.rewardFunc' + str(rCount)), 'reward'))
    except:
        break


nShip = 2 #Number of ships
graphLen = 200 #Number of nodes in the grid
outDeg = 9  # change the outdegree here

graphLink = -np.ones((graphLen, outDeg), dtype=np.int32)
graphDist = -np.ones((graphLen, outDeg), dtype=np.int32)
graphMax = np.zeros(graphLen, dtype=np.int32)

shipPos = np.zeros(nShip, dtype=np.int32) #ships' positions initialization
maxSpeed = np.array([5]*nShip) #Set max speed here

fileNameForGrid = "inputs/200nodes_441edges_degree9.csv" #Name of csv file containing grid

def loadGraph():  # load graph from csv and setup for training 
    global outDeg
    rawData = []
    with open(fileNameForGrid) as csvfile:
        rawData = list(csv.reader(csvfile, delimiter=','))
    rawData.pop(0)

    retLink = -np.ones((graphLen, outDeg), dtype=np.int32)
    retDist = -np.ones((graphLen, outDeg), dtype=np.int32)
    retMax = np.full(graphLen, outDeg - 1, dtype=np.int32)
    rawPoints = -np.ones(graphLen, dtype=np.int32)
    n = 0

    # csv to array
    for point in rawData:
        retMax[n] = outDeg
        j = 0
        for i in range(0, outDeg):
            if (point[i * 2 + 3] == 'N/A' or point[i * 2 + 3] == ''):
                break
            retLink[n][j] = int(point[i * 2 + 3])
            retDist[n][j] = int(point[i * 2 + 3 + 1])
            j += 1
        retMax[n] = j

        rawPoints[n] = point[0]
        n += 1
        
    # init links
    for i in range(0, len(rawPoints)):
        for j in range(0, outDeg):
            if (retLink[i][j] == -1):
                break
            retLink[i][j] = np.where(rawPoints == retLink[i][j])[0]
   

    return retLink, retDist, retMax


def randomLoc():  # generates random ship position
    global outDeg

    retPos = np.zeros(nShip, dtype=np.int32)
    for i in range(0, nShip):
        retPos[i] = randint(0, graphLen - 1)
    return retPos



def initShip():  # init P table and ships
    global outDeg
    shipPos = np.zeros(nShip, dtype=np.int32)
    shipPos = randomLoc()

    pTable = np.full(((nShip, nShip) + ((graphLen,) * nShip) + (outDeg + 1, outDeg + 1)), 1 / (outDeg + 1),
                     dtype=np.float64)
    
    return shipPos, pTable


def initQ():  # init Q table
    global outDeg,graphLen,maxSpeed


    #Shape of Q Arr
    shapeOfArr = (graphLen,) * nShip + (outDeg+1,outDeg+1) * 2 + (maxSpeed[0],)* nShip



    qTable = np.full( shapeOfArr ,1 / (outDeg + 1), dtype=np.float32)
    return qTable



def actionToPos(pos, action, graphLink, graphDist, graphMax):  # convert action taken to a position state on the grid 
    global outDeg
    if (action == outDeg):
        return pos, 0
    return graphLink[pos][action], graphDist[pos][action]


def shipProximity(shipPos, graphLink, graphMax, maxNotSeen, ship, i): #Determine proximity between ships 
    global outDeg

    for p in range(0, outDeg):
        maxNotSeen[ship, i, :, p] = 0
        if p < graphMax[shipPos[i]]:
            for j in range(0, nShip):
                if j != i:
                    if np.any(graphLink[p] == shipPos[j]):
                        maxNotSeen[ship, i, :, p] = 1



def normalTrain(graphLink, graphDist, graphMax, pTable, qTable, qErr, pErr, reward, v, increase, state,a):  # Train with action selection and P table
    global outDeg
    B = np.random.randint(20, 30, nShip) / 100000
    Arr = [0]
    qErrTemp = np.zeros(2, dtype=np.float32)
    pErrTemp = np.zeros(2, dtype=np.float32)
    nQ = 0
    nP = 0
    contFlag = True
    state = np.array([0, 1])
    
    #Perform train for each state pair 
    while state[0] < graphLen:
        actionOut = np.full((nShip, nShip), outDeg)
        speedOut = np.full((nShip, nShip), outDeg)
        SoA = np.full(nShip, outDeg)
        newState = np.full(nShip, 0)
        qArr = np.random.randint(0, outDeg, (nShip, outDeg + 1, outDeg + 1, 2))
        maxNotSeen = np.random.randint(0, outDeg, (nShip, nShip, outDeg + 1, outDeg))
        notExploredDir = np.random.randint(0, 2, (nShip, outDeg))
        goalDir = np.random.randint(-32, outDeg, nShip)
        for i in range(0, nShip):
            stateOut = np.full(nShip, outDeg)
            for p in range(1, outDeg + 1):
                qArr[i, :, p, :] = 0
            countJ = np.array([0, 0])
            for j in range(0, nShip):
                if (i != j):
                    maxVal = -1000
                    shipProximity(state, graphLink, graphMax, maxNotSeen, i, j)
                    pVal = -1000
                    for action in range(0, graphMax[state[j]] + 1):
                        if action == graphMax[state[j]]:
                            action = outDeg
                        pTemp = 0.0

                        # Printing out p table values
                        pTemp = pTable[i][j][state[0]][state[1]][action][maxNotSeen[i][j][action][0]]
                        if (state[0] == 3 and state[1] == 4 and action == 0):
                            print(pTemp)

                        #Find action with highest p value    
                        if pTemp > pVal:
                            pVal = pTemp
                            actionOut[i][j] = action
                            countJ[j] = maxNotSeen[i][j][action][0]

                    stateOut[j], _ = actionToPos(state[j], actionOut[i][j], graphLink, graphDist, graphMax)

            qVal = -1000
            actionJ = actionOut[i].copy()

            #Determine q value for action and speed pairs 
            for action in range(0, graphMax[state[i]] + 1):
                if action == graphMax[state[i]]:
                    action = outDeg
                countJ[i] = qArr[i][action][0][0]
                actionJ[i] = action


                #Try all speeds for asset1
                for speed1 in range(maxSpeed[i]):

                    #Try all speeds for asset2
                    for speed2 in range(maxSpeed[j]):

                        qTemp = qTable[state[0]][state[1]][actionJ[0]][countJ[0]][actionJ[1]][countJ[1]][speed1][speed2]
              

                        actionOut[i][i] = action
                        speedOut[i][i] = speed1
                        speedOut[j][j] = speed2


                        avg = 0


                        #COmpute average of 3 rewards to compute q value 
                        for m in range(len(mod)):
                            currReward = mod[m]
                            newState[i], _ = actionToPos(state[i], actionOut[i][i], graphLink, graphDist, graphMax)
                            SoA[i] = qArr[i, actionOut[i][i]][0][0]
                            qOut = currReward(SoA, newState, nShip, actionOut, speedOut, notExploredDir, goalDir,outDeg)
                            avg  += qOut

                        avg = avg/len(mod)

                        
                        #Store average in q table 
                        qTable[state[0]][state[1]][actionOut[0][0]][qArr[0][actionOut[0][0]][0][0]][
                            actionOut[1][1]][
                            qArr[1][actionOut[1][1]][0][0]][speed1][speed2] *= 1 - a
                        qTable[state[0]][state[1]][actionOut[0][0]][qArr[0][actionOut[0][0]][0][0]][
                            actionOut[1][1]][
                            qArr[1][actionOut[1][1]][0][0]][speed1][speed2] += a * avg







        #Store all P values in P table
        for i in range(0, nShip):
            for j in range(0, nShip):
                if i != j:
                    for action in range(0, graphMax[state[j]] + 1):
                        if action == graphMax[state[j]]:
                            action = outDeg
                        if (action == actionOut[j][j]):
                            pTable[i][j][state[0]][state[1]][actionOut[j][i]][maxNotSeen[i][j][actionOut[j][i]][0]] += \
                            pTable[i][j][state[0]][state[1]][action][maxNotSeen[i][j][action][0]] * B[i]
                        else:
                            pTable[i][j][state[0]][state[1]][action][maxNotSeen[i][j][action][0]] *= (1 - B[i])


        #Move to next state, until all state combinations are tried out 
        state[1] += 1
        if state[1] == graphLen:
            state[0] += 1
            state[1] = 0
        

    return pTable, qTable, qErrTemp, pErrTemp, state


if __name__ == "__main__":


    print("Performing training for Exact MaMoRL. Be aware Q and P tables generated easily be very large....")

    # Init
    print("loading graph")
    graphLink, graphDist, graphMax = loadGraph()
    print("creating p table")
    shipPos, pTable = initShip()
    print("creating q table")
    qTable = initQ()
    

    epoch = 10 #Number of epoch training sessions
    average = 0
    state = np.zeros(nShip, dtype=np.int32)
    a = 0.001

    # Normal Train Q and P tables
    qErrT = 1
    flag = True
    rng = default_rng()
    total_time = 0.0
    ct = datetime.datetime.now()
    print("start  time:-", ct)
    
    for i in range(1, epoch + 1):
        print("epoch -- " + str(i))
        print("START", i)
        print("a:", a)
        start = int(round(time.time())) / 60
        breakVal = False
        for v in range(0, 1):
            pTable, qTable, qErr, pErr, state = normalTrain(graphLink, graphDist, graphMax, pTable, qTable, 1000, 1000, mod[v], v, 1, state, a)
   
            print("Q ERROR - vector " + str(v) + ":", qErr)
            print("P ERROR - vector " + str(v) + ":", pErr)
    

        end = int(round(time.time())) / 60
        total_time += end - start
        average = (average * (i / (i + 1))) + ((end - start) / (i + 1))
        timeRem = (epoch - i) * average
        print(i, "Time Remaining:", timeRem)
 

    st = datetime.datetime.now()
    print("end  time:-", st)

    print("time for training : " + str(st-ct))
    
    print(getsizeof(qTable))
    print(getsizeof(pTable))

    #Save q table -- warning might be very large for mid sized grids 
    np.save("data/qData.npy", qTable)


    #Save p table -- warning might be very large for mid sized grids 
    for i in range(0, nShip):
        for j in range(0, nShip):
            if (i != j):
                np.save("data/ " + str(i) + '-' + str(j) + '-' + 'pData', pTable[i][j])

