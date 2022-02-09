import csv
import datetime
import numpy as np
from random import randint
import time
import importlib
import networkx as nx



# Change below for desired values
nShip = 3 #Put number of ships here
maxSpeed = np.array ([5]*nShip) #Set the max speed here, np array size must equal number of ships
epoch = 10 #number of epoch to train model
a = 0.001
B = np.random.randint(80, 90, nShip) / 10000
graphLen = 40  # Must keep low for better training
outDeg = 9 #Set outdegree here
fileName= "input/400nodes_846edges_degree9.csv" #Set input grid file here
noNodes=400 #number of nodes in the grid
region=240 #number of nodes in the known region. Keep it 60% of noNodes

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
    with open(filename) as csvfile:
        rawData = list(csv.reader(csvfile, delimiter=','))

    #LOAD each line of csv one by one, remove first line as its the headers containing column heading for line. Put in list.
    #Each element in list is one node's information
    rawData.pop(0)
    retLink = -np.ones((len(rawData), outDeg), dtype=np.int64)  # Links from a node
    retDist = -np.ones((len(rawData), outDeg), dtype=np.int64)  # Weights of Links
    retMax = np.full(len(rawData), outDeg-1, dtype=np.int64)  # Outdegree of a node
    rawPoints = -np.ones(len(rawData), dtype=np.int64)  # Storing the nodes in raw node ID
    lat_long = {i: [] for i in range(noNodes)} #store nodes lat/long

    n = 0
    k=0

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
        lat_long[k].append(float(point[1]))
        lat_long[k].append(float(point[2]))
        k += 1


    for i in range(0, len(rawPoints)):
        for j in range(0, outDeg):
            if (retLink[i][j] == -1):
                break
            retLink[i][j] = np.where(rawPoints == retLink[i][j])[0]

    global graphLen #Set above

    return retLink, retDist, retMax, lat_long



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



def normalTrain(graphLink, graphDist, graphMax, pTable, qTable, qErrTemp, pErrTemp, reward, v, a, B, epoch):  # Train with action selection and P table

    global outDeg
    global seenNodes

    nQ = 0.0
    nP = 0.0
    nActions = outDeg + 1 # maximum number of actions
    maxOutDegree = outDeg
    iters = 5000    # Set number of iterations per epoch here

    qValRet = np.zeros(iters * nShip)
    qErrRet = np.zeros(iters * nShip)
    pValRet = np.zeros(iters * nShip * nShip)
    pErrRet = np.zeros(iters * nShip * nShip)

    for iter in range(0, iters):

        # maximum number of actions
        state = np.random.choice(graphLen,nShip)  # Random Sample states
        randomOutDegree = np.random.randint(0, nActions, (nShip, nActions, maxOutDegree + 1, 2))  # Randomize outdegree and store ship crash directions
        notExploredDir = np.random.randint(0, 2, (nShip, nActions - 1))  # Direction to Nearest edge node
        goalDir = np.random.randint(-32, nActions - 1,nShip)  # Direction to goal if sensed (-32 to nActions to give chance of 6 / 48)

        # Initialize for storing
        actionOut = np.full((nShip, nShip), nActions - 1)  # Store actions
        speedOut = np.full((nShip, nShip), nActions - 1)  # Store speed actions
        actionOutSensed = np.full(nShip, nActions - 1)  # Store Outdegree values for actions
        newState = np.full(nShip, 0)  # store state calculated by p table
        shipProximityP = np.random.randint(0, nActions - 1, (nShip, nShip, maxOutDegree + 1, outDeg + 2))  # store ship proximity values for P Array
        shipProximityQ = np.zeros(nShip)  # Store Q Proximity

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
                        qArrOut[j + 4] = getShipProximityQ(graphLink[state[i]][action], state, graphLink, graphMax, i,1)  # proximity
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

            newState[i], _ = actionToPos(state[i], actionOut[i][i], graphLink, graphDist, graphMax)
            actionOutSensed[i] = randomOutDegree[i, actionOut[i][i]][0][0]

        if epoch >= 9:
            speedOut = np.random.randint(0, nActions, (nShip, nShip))  # Store speed actions

        # get reward and recalculate weights
        for i in range(0, nShip):
            # get reward
            qOut = reward(actionOutSensed, newState, nShip, actionOut, speedOut, notExploredDir, goalDir, shipProximityQ,outDeg, seenNodes)
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


                    Err = pOut - np.sum((pTable[i][j][v][0] * pArrOut))

                    if v == 0:
                        pErrRet[iter * nShip * nShip + i * nShip + j] = np.sum((pTable[i][j][v][0] * pArrOut))

                    if actionOut[i][j] != outDeg:
                        pErrTemp[0] += Err ** 2  # error for action
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

    # init
    loadModules()  #Loads the reward function from each reward module we provide
    graphLink, graphDist, graphMax,  lat_long = loadGraph() #load graph

    # Convert graph format using the NetworkX library
    G = nx.Graph()
    G.add_nodes_from([i for i in range(noNodes)])

    edges = []
    for i in G.nodes():
        for j in graphLink[i]:
            if (j != -1):
                edges.append((i, j))

    G.add_edges_from(edges)

    weights = {}
    for edge in G.edges():
        weights[edge] = graphDist[edge[0]][np.where(graphLink[edge[0]] == edge[1])[0][0]]
        weights[edge[::-1]] = graphDist[edge[0]][np.where(graphLink[edge[0]] == edge[1])[0][0]]

    for e in G.edges():
        G[e[0]][e[1]]['weight'] = weights[e]

    nx.set_node_attributes(G, lat_long, "lat/long")


    # Assign a connected sungrid as the known region that contains destination
    def subGrid(G, region):

        degrees = {}
        for node, val in G.degree():
            degrees[node] = val
        degrees = {k: v for k, v in sorted(degrees.items(), key=lambda item: item[1], reverse=True)}
        selected_nodes = []
        i = 0
        for key in degrees:
            selected_nodes.append(key)
            i += 1
            if i == region:
                break

        SG = G.subgraph(selected_nodes)

        maxLen = 0
        if not nx.is_connected(SG):
            S = [SG.subgraph(c).copy() for c in nx.connected_components(SG)]
            for g in S:
                if len(list(g.nodes())) > maxLen:
                    maxLen = len(list(g.nodes()))
                    SG1 = g

        else:
            SG1 = SG
            maxLen = len(SG1.nodes())

        return SG1, maxLen

    # Return number of nodes in the known region
    SG, maxLen = subGrid(G, region)
    print("length of subgrid=", maxLen)

    # nodes in the grid that ships sensed them
    seenNodes = list(set(G.nodes) - set(SG.nodes))

    pTable = initPTable()  #  P Table initialized
    qTable = initQ()  # initializes Q Table

    # Normal Train Q and P tables
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
            pErr = np.zeros(2, dtype=np.float64)
            pTable, qTable, nQ, nP, qValRet, qErrRet, pValRet, pErrRet = normalTrain(graphLink, graphDist, graphMax, pTable, qTable, qErr, pErr, mod[v], v, a, B, i)

            qErr = qErr / nQ  # calculate error
            pErr = pErr / nP
            print(qErr,pErr)
            print(nQ,nP)
            print("Q ERROR - vector " + str(v) + ":", qErr)
            print("P ERROR - vector " + str(v) + ":", pErr)
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
    for i in range(0, nShip):
        for j in range(0, nShip):
            if (i != j):
                np.save("data/ " + str(i) + '-' + str(j) + '-' + 'pData', pTable[i][j])

