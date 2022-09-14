from random import randint
import numpy as np
import itertools as it
import csv
import time
import math


"""
Exact MaMoRL -- Run File
Train before using 

"""


noOfShip,deg = [],[] #To run different values so it will run for each instance automatically or just one value for a particular run
noOfShip = [2]  #Set number of ships 
deg=[9] #Set outdegree of grid 
noOfIter = 10 #The number of times to run the code for diffwrwnt starting points and destinations
sendInterval = 1 #ships' sensing raduis
goalSensed = False
seen = []
fileName='inputs/200nodes_441edges_degree9.csv'  #input grid directory
resultPath='result.csv' #directory and fileanme to store results
noNodes = 200 #number of nodes in the grid
cFreq = 3 # communication interval


rawTest = []
class node:

    def getPosition(self):  # get adjacant nodes
        return self.pos

    def addPossible(self, addDict):  # add adjacant nodes
        self.possibleNodes.update(addDict)

    def getDistance(self, node):  # get distance to adjascant node
        return self.possibleNodes[node]

    def __init__(self, addPos, loc, index, minMax):  # init node members
        self.possibleNodes = dict()
        self.pos = index
        self.rawPos = addPos
        self.loc = loc




class Ship:
    def getPossibleActions(self, Graph, ships):  # get possible actions from node
        actions = ()
        tempAction = ()
        a = 0
        for action in Graph[self.pos].possibleNodes.keys():
            check = False
            for ship in ships:
                if ship.Num != self.Num:
                    if ship.pos == action and ship.pos != ship.dest:
                        check = True
                        break

            if not check:
                tempAction = tempAction + (a,)
                actions = actions + (action,)
                a += 1
        tempAction = tempAction + (outDeg,)
        actions = actions + (ships[self.Num].pos,)
        return actions, tempAction

    def randLoc(self, Graph, iter):  # randomize ship location
        self.pos = randint(0, len(Graph) - 1)
        self.time = 0


    def __init__(self, shipNo, Graph):  # initialize ship members
        global goalNode
        self.Num = shipNo
        self.randLoc(Graph, 0)
        self.dest = goalNode  #Set the destination for ships here
        self.pTable = dict()
        self.time = 0
        self.speed = 0  
        self.maxSpeed = 5
        self.oldPos = self.pos
        self.movPos = [0, 0]
        self.fuel = 0
        self.otherPos = 0
        self.seen = []



def loadGraph():  # get graph from csv to use for running 
    global outDeg
    rawData = []
    global rawTest
    with open(fileName) as csvfile:  # Set name of input grid here
        rawData = list(csv.reader(csvfile, delimiter=','))
    rawData.pop(0)

    for point in rawData:
        rawTest.append(int(point[0]))

    xPoints = []
    yPoints = []
    xMax = rawData[0][1]
    xMin = rawData[0][1]
    yMax = rawData[0][2]
    yMin = rawData[0][2]
    for points in rawData:
        if (float(points[1]) not in xPoints):
            xPoints.append(float(points[1]))
        if (float(points[2]) not in yPoints):
            yPoints.append(float(points[2]))

    xMax = max(xPoints)
    xMin = min(xPoints)
    yMax = max(yPoints)
    yMin = min(yPoints)
    Graph = []

    minMax = (xMin, yMin, xMax, yMax)
    n = 0
    graphPoints = []
    outPoints = []
    
    for point in rawData:

        nodePossible = dict()
        for i in range(0, outDeg):
            if (point[i * 2 + 3] == 'N/A' or point[i * 2 + 3] == ''):
                break
            if int(point[i * 2 + 3]) in rawTest:
                nodePossible.update({int(point[i * 2 + 3]): int(point[i * 2 + 1 + 3])})
        if (int(point[0]) in rawTest):
            tempNode = node(int(point[0]), (float(point[1]), float(point[2])), n, minMax)
            tempNode.addPossible(nodePossible)
            Graph.append(tempNode)
            graphPoints.append(int(point[0]))
            n += 1
    Graph = np.array(Graph)
    for i in range(0, len(Graph)):
        newPossible = dict()
        for connPoint in Graph[i].possibleNodes.keys():
            outPoint = graphPoints.index(connPoint)
            newPossible.update({outPoint: Graph[i].possibleNodes[connPoint]})

        Graph[i].possibleNodes = newPossible

    return Graph


def djikstra(pos, Graph, shipSeen):  # djikstra algorithm
    global outDeg
    Q = []
    dist = []
    prev = []
    corners = dict()
    for j in range(0, len(Graph)):
        if j in shipSeen:
            Q.append(j)
        dist.append(1000000)
        prev.append([])
    dist[pos] = 0
    prev[pos] = [pos]
    while (Q):
        u = Q[0]
        for j in Q:
            if (dist[j] < dist[u]):
                u = j

        if (detAdjNum(Graph, u, shipSeen) < len(Graph[u].possibleNodes.keys())):
            corners.update({u: dist[u]})

        Q.remove(u)
        for i in Graph[u].possibleNodes.keys():
            v = i
            if (v in shipSeen):
                alt = dist[u] + Graph[u].possibleNodes[v]

                if (alt < dist[v]):
                    dist[v] = alt
                    tempPrev = prev[u].copy()
                    tempPrev.append(v)
                    prev[v] = tempPrev

    return corners, dist, prev



def initShips(Graph):  # init ship objects

    global outDeg

    ships = []
    pTemp = (np.full((len(Graph), len(Graph), outDeg + 1, outDeg), (1 / (outDeg + 1))))  # initialize p table
    for i in range(0, nShip):
        ship = Ship(i, Graph)
        pTable = dict()
        for j in range(0, nShip):
            if (i != j):
                pTable.update({j: pTemp})
        ship.pTable = pTable
        ships.append(ship)
    ships = np.array(ships)
  
    return ships


def detAdjNum(Graph, n, shipSeen): #Determine number of sensed nodes 
    global outDeg
    count = 0
    for i in Graph[n].possibleNodes.keys():
        
        if i in shipSeen:
            count += 1
    return count


def initQTable(Graph):  # init Q variable
    global outDeg
    qTable = []
    qTable = np.full((((len(Graph),) * nShip) + ((outDeg + 1,) * nShip)), (1 / ((outDeg + 1) ** nShip)))
    return qTable


def reloadShips(ships, Graph, iter):  # randomize ships
    global outDeg
    for i in range(0, nShip):
        ships[i].randLoc(Graph, iter)
    return ships


def collisiontDetect(ships):  # if collision occurs,detect it and return True, else False 
    global outDeg
    for i in range(0, nShip):
        for j in range(i, nShip):
            if (i != j and ships[i].pos == ships[j].pos and ships[i].pos != ships[i].dest and ships[j].pos != ships[
                j].dest):
                return True
    return False


def endDet(ships):  # if all ships reach goal
    global outDeg
    for ship in ships:
        if (ship.pos != ship.dest):
            return False
    return True



def getPossible(ships, pos, i, Graph):  # possible actions from node
    global outDeg
    outputNodes = []
    for node in Graph[pos].possibleNodes:
        check = True
        for j in range(0, 2):
            if (i != j):
                if (ships[j].pos == node and ships[j].dest != ships[j].pos):
                    check = False
        if (node < 0 and node > len(Graph)):
            check = False
        if (check):
            outputNodes.append(node)
    return outputNodes


def actionToPos(pos, action, Graph):  # convert action to node position
    global outDeg
    if (action == outDeg):
        return pos, 0
    return list(Graph[pos].possibleNodes.keys())[action], list(Graph[pos].possibleNodes.values())[action]


tracker1 = []
tracker2 = []


def calcNewPos(ships, Graph, qTable, t, test=1):  # Take action based on q and p tables 
    global outDeg
    global goalSensed, tracker1, tracker2
    if test < 0:
        a = 0.6
    state = ()
    actionComb = []
    posOut = []
    if test > 0:
        Tb = 0
    for ship in ships:
        state = state + (ship.pos,)
        posOutTemp, actionTemp = ship.getPossibleActions(Graph, ships)
        posOut.append(posOutTemp)
        actionComb.append(actionTemp)
    actionOut = [outDeg, outDeg]
    for i in range(0, nShip):

        if (ships[i].pos == ships[i].dest):
            actionOut[i] = outDeg
            ships[i].speed = 0
            continue

        if (ships[i].pos != ships[i].dest):
            state = [0, 0]
            state[i] = ships[i].pos
            for j in range(0, nShip):
                if i != j:
                    if t % cFreq == 0:
                        state[j] = ships[j].pos
                        ships[i].otherPos = ships[j].pos
                    state[j] = ships[i].otherPos
                    otherComb = np.arange(len(Graph[ships[i].otherPos].possibleNodes.keys()) - 1).tolist()
                    otherComb.append(outDeg)
            actionJ = [outDeg, outDeg]
            countJ = [0, 0]
            speedJ = [4,4] #set here speed
            speedOut = [0,0]
            nearCorner = False
            # get max P
            for j in range(0, nShip):
                if (i != j):
                    pVal = -1000
                    for action in otherComb:
                        count = detAdjNum(Graph, actionToPos(state[j], action, Graph)[0], ships[j].seen)
                        pTemp = ships[i].pTable[j][ships[i].pos][state[j]][action][count]  
                        if pTemp > 0:
                            nearCorner = True
                        if (pTemp > pVal):
                            pVal = pTemp
                            actionJ[j] = action
                            countJ[j] = count
                    if (state[j] == ships[j].dest):
                        actionJ[j] = outDeg

                    if goalSensed:
                        _, dist, prev = djikstra(state[j], Graph, ships[j].seen)
                        nextPos = prev[ships[j].dest]
                        nextAction = outDeg
                        try:
                            nextAction = list(Graph[state[j]].possibleNodes.keys()).index(nextPos[1])
                        except:
                            None
                        countJ[j] = detAdjNum(Graph, actionToPos(state[j], nextAction, Graph)[0], ships[j].seen)
                        actionJ[j] = nextAction

                    if (nearCorner == False) and state[j] != ships[j].dest and (
                            not goalSensed or dist[ships[j].dest] == 1000000):
                        corners, _, prev = djikstra(state[j], Graph, ships[j].seen)
                        nextPos = prev[min(corners, key=corners.get)]
                        nextAction = outDeg
                        
                        try:
                            nextAction = list(Graph[state[j]].possibleNodes.keys()).index(nextPos[1])
                        except:
                            None
                        countJ[j] = detAdjNum(Graph, actionToPos(state[j], nextAction, Graph)[0], ships[j].seen)
                        actionJ[j] = nextAction
                        
                    ships[i].otherPos, _ = actionToPos(ships[i].otherPos, actionJ[j], Graph)
            qVal = -10000
            nearCorner = False
            actionJ[i] = 0
            countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, 0, Graph)[0], ships[i].seen)
         
            zhy = (state[0], state[1], actionJ[0], countJ[0], actionJ[1],
                   countJ[1],speedJ[0],speedJ[1])  # Gets the qval based on the way it is stored in training
            qMat = zhy
            
            base = qTable[qMat]
            if base < 0:
                actionJ[i] = 1
                countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, 1, Graph)[0], ships[i].seen)
                zhy = (state[0], state[1], actionJ[0], countJ[0], actionJ[1],
                       countJ[1],speedJ[0],speedJ[1])  # Gets the qval based on the way it is stored in training
                qMat = zhy
    
                base = qTable[qMat]


            # get max V
            for action in actionComb[i]:
                actionJ[i] = action
                countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, action, Graph)[0], ships[i].seen)
               

               
                #Try for best speed pair for the ships 
                for speed1 in range(speedJ[0]):

                    for speed2 in range(speedJ[1]):

                        
                        zhy = (state[0], state[1], actionJ[0], countJ[0], actionJ[1],
                               countJ[1],speed1,speed2)  # Gets the qval based on the way it is stored in training
                        qMat = zhy
                        qTemp = qTable[qMat]
                        if qTemp > 0 and qTemp != base:
                            nearCorner = True
                        if (qTemp >= qVal):
                            qVal = qTemp
                            actionOut[i] = action
                            speedOut[i] = speed1
                            speedOut[j] = speed2

            if goalSensed:
                _, dist, prev = djikstra(ships[i].pos, Graph, ships[i].seen)
                if (dist[ships[i].dest] != 1000000):
                    nextPos = prev[ships[i].dest]
                    nextAction = outDeg
                    try:
                        nextAction = list(Graph[ships[i].pos].possibleNodes.keys()).index(nextPos[1])
                    except:
                        None
                    countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, nextAction, Graph)[0], ships[i].seen)
                    actionJ[i] = nextAction
                    actionOut[i] = nextAction

            if (nearCorner == False) and (not goalSensed or dist[ships[i].dest] == 1000000):
                corners, _, prev = djikstra(ships[i].pos, Graph, ships[i].seen)
                crash = -1

                while crash == -1:
                    nextAction = outDeg
                   

                    try:
                        minOpen = min(corners, key=corners.get)
                        nextPos = prev[minOpen]
                        nextAction = list(Graph[ships[i].pos].possibleNodes.keys()).index(nextPos[1])
                    except:
                        None
                    countJ[j] = detAdjNum(Graph, actionToPos(ships[i].pos, nextAction, Graph)[0], ships[i].seen)

                    actionOut[i] = nextAction
                    actionJ[i] = nextAction
             
                    zhy = (state[0], state[1], actionJ[0], countJ[0], actionJ[1],
                           countJ[1],speedOut[0],speedOut[1])  # Gets the qval based on the way it is stored in training
                    qMat = zhy
             
                    crash = qTable[qMat]
                    try:
                        corners.pop(minOpen)
                    except:
                        None
                   

            actionJ[i] = actionOut[i]
           
            if actionOut[i] == outDeg:
                ships[i].speed = 0
            else:
                ships[i].speed = speedOut[0]
                ships[i].speed = speedOut[1]

    maxTime = 0
    for i in range(0, nShip):
        
        ships[i].oldPos = ships[i].pos
        newPos, timeTemp = actionToPos(ships[i].pos, actionOut[i], Graph)

        if ships[i].speed != 0:

            timeTemp = timeTemp / ships[i].speed
            
      
        ships[i].pos = newPos
        ships[i].time += timeTemp
       
        if ships[i].pos != ships[i].oldPos:
            ships[i].fuel += timeTemp * (
                        0.2525 * math.pow((ships[i].speed + 10) * 2, 2) - 1.6307 * (ships[i].speed + 10) * 2)
           
        if (timeTemp > maxTime):
            maxTime = timeTemp

        for j in Graph[ships[i].pos].possibleNodes.keys():
            if j not in ships[i].seen:
                ships[i].seen.append(j)
        for j in Graph[ships[i].otherPos].possibleNodes.keys():
            if j not in ships[i].seen:
                ships[i].seen.append(j)
        seen = ships[0].seen.copy()
        if ships[i].dest in ships[i].seen:
            goalSensed = True
            for j in range(0, nShip):
                if i != j and ships[j].dest not in ships[j].seen:
                    ships[j].seen.append(ships[j].dest)
       
    for i in range(0, nShip):
        if (ships[i].pos == ships[i].oldPos and ships[i].pos != ships[i].dest):
            ships[i].time += maxTime

    t += 1
    return qTable, ships, t


def printMaxTime(ships):  # print time at end

    global outDeg, maxTime, fuels, times

    maxTime = 0



    times = []
    fuels = []
    # calc max values
    for ship in ships:
        if (ship.time > maxTime):
            maxTime = ship.time

        fuels.append(ship.fuel)
        times.append(ship.time)

    max1 = round(maxTime, 4)

    print("Times for n ships : " + str(times))
    print("Fuels for n ships : " + str(fuels))
    print()

    print("Max time for n ships is: " + str(maxTime))
    print("Sum of all fuel is : " + str(sum(fuels)))

    return maxTime


ts = 0
te = 0


def mainFunc():  # Main function
    global outDeg
    global goalSensed
    global seen
    global ts, te
    global qSize, pSize

    # init
    Graph = loadGraph()

 
    ships = initShips(Graph)

    qTable = initQTable(Graph)
    # load data
    try:
        qTable = np.load("qData.npy")

        # https://www.geeksforgeeks.org/find-the-memory-size-of-a-numpy-array/
        print("Q Array size is " + str(qTable.itemsize * qTable.size) + " bytes")
        qSize=  str(qTable.itemsize * qTable.size)
        totalPTableSize = 0

        for i in range(0, 2):
            pTemp = dict()
            for j in range(0, 2):
                if (i != j):
                    pArray = np.load(str(i) + '-' + str(j) + '-' + 'pData.npy')
                    totalPTableSize += pArray.itemsize * pArray.size
                    pTemp.update({j: pArray})
            ships[i].pTable = pTemp
        qTable = np.array(qTable)
        print("Total P table size is " + str(totalPTableSize) + " bytes")
        pSize=str(totalPTableSize)
    except:
        print("NO DATA")

 
    crashed = False
    t = 0
    iter = 0
    iterTime = []

    global seen
    seen = []

    for i in range(0, len(ships)):
        seen.append(ships[i].pos)
        for j in Graph[ships[i].pos].possibleNodes.keys():
            seen.append(j)
    for i in range(0, len(ships)):
        ships[i].seen = seen.copy()
    iterTime = 0
    while not crashed:
        xChange = 1
        refresh = 0

        # if end or collision
        if refresh or t > 500 or collisiontDetect(ships):
            print("COLLIDED")
            ts = time.time()
            goalSensed = False
            seen = []

            exit()
            # refresh positions
            reloadShips(ships, Graph, 0)
            for i in range(0, len(ships)):
                seen.append(ships[i].pos)
                for j in Graph[ships[i].pos].possibleNodes.keys():
                    seen.append(j)
            for i in range(0, len(ships)):
                ships[i].seen = seen.copy()
            t = 0
            iter += 1
        if endDet(ships):
            seen = []
            goalSensed = False

            iterTime += printMaxTime(ships)
            print("All at goal")
            print()
            print("ship locations : " + str(ships[0].pos) + "," + str(ships[1].pos) + "," + str(Graph[ships[1].pos].loc))
            te = time.time()
            print("Code ran for : " + str(te - ts))
            return 0
            exit()

            time.sleep(10)
            reloadShips(ships, Graph, iter)
            for i in range(0, len(ships)):
                seen.append(ships[i].pos)
                for j in Graph[ships[i].pos].possibleNodes.keys():
                    seen.append(j)
            for i in range(0, len(ships)):
                ships[i].seen = seen.copy()
            t = 0
            iter += 1

        if (xChange > 0):
            # take action and update tables and ship locations 

            qTable, ships, t = calcNewPos(ships, Graph, qTable, t, 1)

        if iter == 3305:
            break


if __name__ == "__main__":


    #Automation code to allow several runs in one run 
    for _ in range(noOfIter):
        for i in noOfShip:
            for j in deg:

                nShip = i  # number of ships here
                outDeg = j # the MAX outdeg here
                goalNode = randint(0,noNodes) # randomly select a node index as the destination. ''' You can fix the nodes index as well'''
                failed,count=1,0
                while failed:
                    failed=mainFunc()
                    count += 1 #keep track of number of collisions
                with open(resultPath, mode='a') as csv_file:
                    fieldnames = ['No of Ships', 'Out-degree', 'Destination Index','No of Times collided',  'Max time', 'Total Fuel', 'Running time', 'qSize', 'pSize']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    if csv_file.tell() == 0:
                        writer.writeheader()
                    writer.writerow({'No of Ships': nShip, 'Out-degree': outDeg, 'Destination Index': goalNode,'No of Times collided': count-1,
                    'Max time':str(maxTime), 'Total Fuel':str(sum(fuels)), 'Running time':str(te - ts), 'qSize':str(qSize) , 'pSize': str(pSize)})
