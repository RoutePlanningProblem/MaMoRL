from random import randint
import numpy as np
import csv
import time
import math
from decision import speedChoice, actionChoice
from tensorflow import keras


# change
sendInterval = 1 #Ships'sensing raduis
goalSensed = False

"""Set the values here"""
nShip = 2  # Set number of ships here
outDeg = 6 #Set the MAX outdeg here
goalNode = 1 # Set the node index you want as the goal
filename = 'input/50nodes_93Edges_degree6.csv' #Name of file with input grid
cFreq = 3

# Define nodes in rawTest in case using on smaller graph, else leave empty
rawTest = []

class node:

    def getPosition(self):  # get adjacant nodes
        return self.pos

    def addPossible(self, addDict):  # add adjacant nodes
        self.possibleNodes.update(addDict)

    def getDistance(self, node):  # get distance to adjascant node
        return self.possibleNodes[node]


    def __init__(self, addPos, loc, index):  # init node members
        self.possibleNodes = dict()  # links
        self.pos = index  # position
        self.rawPos = addPos
        self.loc = loc



class Ship:
    def getPossibleActions(self, Graph, ships):  # get possible actions from node
        actions = ()
        tempAction = ()
        a = 0

        # checks if there is another ship on given node, then that is not returned as action
        for action in Graph[self.pos].possibleNodes.keys():  # from graph possible nodes

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
        # print(actions)
        return actions, tempAction

    def randLoc(self, Graph):  # randomize ship location
        self.pos = randint(0, len(Graph) - 1)  # position
        self.time = 0  # time
        self.oldPos = self.pos  # for drawing movement



    def assignFixedLocs (self,Graph,loc):
        '''Assign a ship location as a fixed parameter, instead of randomly'''
        self.pos = loc  # position
        self.time = 0  # time
        self.oldPos = self.pos  # for drawing movement


    def __init__(self, shipNo, Graph):  # initialize ship members
        global goalNode
        self.Num = shipNo
        self.randLoc(Graph) # randomly reloc ships
        # self.assignFixedLocs(Graph, loc=startLoc) #uncomment this line for fixed location and comment out the previous line
        self.dest = goalNode  # destination
        self.pTable = dict()
        self.time = 0 #initial time
        self.speed = 5 #initial speed
        self.maxSpeed = 5  # Max speed
        self.oldPos = self.pos
        self.movPos = [0, 0]
        self.fuel = 0
        self.otherPos = np.full(nShip, 0).tolist()  # position of other ships
        self.seen = []



def loadGraph():  # get graph from csv
    rawData = []
    global outDeg
    with open(filename) as csvfile:  #Place grid name here
        rawData = list(csv.reader(csvfile, delimiter=','))
    rawData.pop(0)
    xPoints = []
    yPoints = []


    smallerGraph = False #Set true if wanting smaller subset of graph, rawTest must be defined

    global rawTest
    if not smallerGraph:
        rawTest = []
        for point in rawData:
            rawTest.append(int(point[0]))

    # Store latitude and longitude data (x and y respectively) in lists
    for points in rawData:
        if (float(points[1]) not in xPoints):
            xPoints.append(float(points[1]))
        if (float(points[2]) not in yPoints):
            yPoints.append(float(points[2]))

    # Gets max and min of these x y coordinates
    xMax = min(xPoints)
    xMin = max(xPoints)
    yMax = min(yPoints)
    yMin = max(yPoints)
    Graph = []
    n = 0
    graphPoints = []



    # Load from data to links
    for point in rawData:

        nodePossible = dict()
        for i in range(0, outDeg):
            if (point[i * 2 + 3] == 'N/A' or point[i * 2 + 3] == ''):
                break
            if int(point[i * 2 + 3]) in rawTest:
                nodePossible.update({int(point[i * 2 + 3]): int(point[i * 2 + 1 + 3])})
        if (int(point[0]) in rawTest):
            tempNode = node(int(point[0]), (float(point[1]), float(point[2])), n)
            if float(point[1]) < xMin:
                xMin = float(point[1])
            if float(point[1]) > xMax:
                xMax = float(point[1])
            if float(point[2]) < yMin:
                yMin = float(point[2])
            if float(point[2]) > yMax:
                yMax = float(point[2])
            tempNode.addPossible(nodePossible)
            Graph.append(tempNode)
            graphPoints.append(int(point[0]))
            n += 1
    Graph = np.array(Graph)

    minMax = (xMin, yMin, xMax, yMax)



    # links to nodes 0 - n
    for i in range(0, len(Graph)):
        newPossible = dict()
        for connPoint in Graph[i].possibleNodes.keys():
            outPoint = graphPoints.index(connPoint)
            newPossible.update({outPoint: Graph[i].possibleNodes[connPoint]})

        Graph[i].possibleNodes = newPossible

    return Graph

def djikstra(pos, Graph, shipSeen):  # modified djikstra algorithm
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
            corners.update({u: dist[u]})  # get nearest corner

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
    ships = []

    locs = []
    for i in range(0, nShip):
        ship = Ship(i, Graph)
        ships.append(ship)
        locs.append(ship.pos)
    ships = np.array(ships)
    print("Ship locations are "+ str(locs))
    return ships

'''use the following function for the fixed location scenario, else use the above one '''

# def initShips(Graph):  # init ship objects
#     global startLocs  # Starting location of n ships, defined at top
#     ships = []
#     print("Starting state of n ships: "+ str (startLocs))
#     for i in range(0, nShip):
#         ship = Ship(i, Graph,startLocs[i])
#         ships.append(ship)
#     ships = np.array(ships)
#     return ships

def detAdjNum(Graph, n, shipSeen):  # return outdegree
    count = 0
    for i in Graph[n].possibleNodes.keys():
        if i not in shipSeen:
            count += 1
    return count


def initQTable(Graph):  # init Q variable
    qTable = np.full((((len(Graph),) * nShip) + ((7,) * nShip)), (1 / (7 ** nShip)))
    return qTable


def reloadShips(ships, Graph):  # randomize ship positions
    for i in range(0, nShip):
        reFlag = True
        while reFlag:
            ships[i].randLoc(Graph)
            reFlag = False
            for j in range(0, i):
                if i != j:
                    if ships[j].pos in Graph[ships[i].pos].possibleNodes.keys():
                        reFlag = True
                        break
    return ships


def collisiontDetect(ships):  # if collision occurs
    for i in range(0, nShip):
        for j in range(i, nShip):
            if (i != j and ships[i].pos == ships[j].pos and ships[i].pos != ships[i].dest and ships[j].pos != ships[j].dest):
                return True
    return False


def endDet(ships):  # if all ships reach goal
    for ship in ships:
        if (ship.pos != ship.dest):
            return False
    return True


def printMaxTime(ships):  # print time at end

    global max1, times, fuels

    max1 = 0

    # calc max values
    times = []
    fuels = []
    # calc max values
    for ship in ships:
        if (ship.time > max1):
            max1 = ship.time
        fuels.append(ship.fuel)
        times.append(ship.time)

    max1 = round(max1, 4)

    print("Times for n ships : " + str(times))
    print("Fuels for n ships : " + str(fuels))
    print()
    print("Max time for n ships is: " + str(max1))
    print("Sum of all fuel is : "+ str(sum(fuels)))


    return max1


def actionToPos(pos, action, Graph):  # convert action to node position
    if (action == outDeg):
        return pos, 0

    return list(Graph[pos].possibleNodes.keys())[action], list(Graph[pos].possibleNodes.values())[action]


def shipProximity(shipPos, Graph, maxNotSeen, i, nShip):  # ship proximity for p
    n = 0
    for p in Graph[shipPos[i]].possibleNodes.keys():
        for j in range(0, nShip):
            if j != i:
                if shipPos[j] in Graph[p].possibleNodes.keys():
                    maxNotSeen[n + 1] = 1
        n += 1


def shipProximityQ(pos, shipPos, Graph, i, n, nShip, dest, hopOut):  # ship proximity for q
    if pos == dest:
        return -1
    for p in Graph[pos].possibleNodes.keys():
        for j in range(0, nShip):
            if j != i:
                if shipPos[j] == p and shipPos[j] != dest:
                    return hopOut
            if n > 1:
                hopOutRec = shipProximityQ(p, shipPos, Graph, i, n - 1, nShip, dest, hopOut + 1)
                if hopOutRec:
                    return hopOutRec
                if hopOutRec == -1:
                    return 0
    return 0


def shipsWithinNHops(pos, ships, Graph, i, n, nShip, dest, shipArr):  # ship proximity for q but with n hops
    if pos == dest:
        return -1
    for p in Graph[pos].possibleNodes.keys():
        for j in range(0, nShip):
            if j != i:
                if ships[j].pos == p and ships[j].pos != dest:
                    shipArr.append(j)
            if n > 1:
                hopOutRec = shipsWithinNHops(p, ships, Graph, i, n - 1, nShip, dest, shipArr)
    return 0


def completeDecision(ships, Graph, actionComb, nShip, t):  # calculate actions
    global outDeg

    # Load NN models
    reconstructed_model_q = keras.models.load_model("NN_Models/Neural_Net_q")
    reconstructed_model_p01 = keras.models.load_model("NN_Models/Neural_Net_p01")
    reconstructed_model_p10 = keras.models.load_model("NN_Models/Neural_Net_p10")


    actionOut = np.full(nShip, outDeg).tolist()
    for i in range(0, nShip):
        if (ships[i].pos != ships[i].dest):  # If the position is not the destination
            state = np.full(nShip, 0).tolist()
            otherComb = []
            shipArr = []

            # get nearby ships
            shipsWithinNHops(ships[i].pos, ships, Graph, i, 2, nShip, ships[i].dest, shipArr)

            # get other ships locations every t hops
            for j in range(0, nShip):
                if i != j:
                    if t % cFreq == 0 or j in shipArr:
                        state[j] = ships[j].pos
                        ships[i].otherPos[j] = ships[j].pos
                    state[j] = ships[i].otherPos[j]
                    otherCombTemp = np.arange(len(Graph[ships[i].otherPos[j]].possibleNodes.keys()) - 1).tolist()
                    otherCombTemp.append(outDeg)
                else:
                    otherCombTemp = actionComb[i]
                otherComb.append(otherCombTemp)
            state[i] = ships[i].pos

            # initialize for Teammate model
            actionJ = np.full(nShip, outDeg).tolist()
            speedJ = np.full(nShip, outDeg).tolist()
            countJ = np.full(nShip, 0).tolist()
            nearCorner = False

            # get max P
            for j in range(0, nShip):
                if ships[j].pos != ships[j].dest:
                    if (i != j):
                        pVal = -1000
                        pArr = np.zeros(outDeg + 1)
                        ships[i].otherPos[i] = ships[i].pos
                        shipProximity(ships[i].otherPos, Graph, pArr, j, nShip)  # get proximity to ships

                        # get action to nearest corners
                        corners, _, prev = djikstra(state[j], Graph, ships[j].seen)
                        pClosest = np.zeros(outDeg + 1)
                        for nodes in prev:
                            if nodes != []:
                                try:
                                    pClosest[list(Graph[state[j]].possibleNodes.keys()).index(nodes[1])] = 1
                                except:
                                    None

                        # if goal sensed get djikstra and action to goal
                        goalDir = -1
                        if goalSensed:
                            _, dist, prev = djikstra(state[j], Graph, ships[j].seen)
                            nextPos = prev[ships[j].dest]
                            try:
                                goalDir = list(Graph[state[j]].possibleNodes.keys()).index(nextPos[1])
                            except:
                                None

                        # action of ship j
                        for action in otherComb[j]:

                            count = detAdjNum(Graph, actionToPos(state[j], action, Graph)[0], ships[j].seen)
                            pArr[0] = count / 6
                            if action != outDeg:
                                # ParrOut is the follow:
                                # outdegree, proximity to another ship, closest edge, speed, goal direction
                                pArrOut = [count / 6, pArr[action + 1], pClosest[action], ships[j].maxSpeed, 0]
                                if goalDir == action:
                                    pArrOut[4] = 1
                            else:  # if action == outDeg / stay
                                pArrOut = [0, 0, 0, 0, 0]


                            newState, _ = actionToPos(state[j], action, Graph)
                            arr=[state[i], state[j], newState]+list(pArrOut)
                            arrInt=np.array([int(x) for x in arr])
                            arrInt=arrInt.reshape(1,8)

                            if i==0 and j==1:
                                pTemp = reconstructed_model_p01.predict(arrInt)[0][0]  # get p value from decision function
                            elif i==1 and j==0:
                                pTemp = reconstructed_model_p10.predict(arrInt)[0][0]  # get p value from decision function


                            if pTemp > 0:
                                nearCorner = True
                            if (pTemp > pVal):
                                pVal = pTemp
                                actionJ[j] = action
                                countJ[j] = count
                        if (state[j] == ships[j].dest):
                            actionJ[j] = outDeg
                        pVal = -1000


                    if i != j:
                        # update position of other ships based on p action
                        ships[i].otherPos[j], _ = actionToPos(state[j], actionJ[j], Graph)

            qVal = -10000
            nearCorner = False
            actionJ[i] = 0
            countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, 0, Graph)[0], ships[i].seen)
            qArr = np.zeros((outDeg + 1, 2))
            corners, _, prev = djikstra(state[i], Graph, ships[i].seen)
            qClosest = np.zeros(outDeg + 1)

            # get closest corner
            for nodes in prev:
                if nodes != []:
                    try:
                        qClosest[list(Graph[state[i]].possibleNodes.keys()).index(nodes[1])] = 1
                    except:
                        None

            n = 0
            # update if any collision action
            for p in Graph[ships[i].pos].possibleNodes.keys():
                for j in range(0, nShip):
                    if ships[j].pos != ships[j].dest:
                        if i != j:
                            if ships[i].otherPos[j] == p and p != ships[i].dest:
                                qArr[n + 1, 0] = 1
                                qArr[n + 1, 1] = speedJ[j]
                n += 1

            goalDir = -1
            # get goal direction
            if goalSensed:
                _, dist, prev = djikstra(ships[i].pos, Graph, ships[i].seen)
                if (dist[ships[i].dest] != 1000000):
                    nextPos = prev[ships[i].dest]
                    try:
                        goalDir = list(Graph[ships[i].pos].possibleNodes.keys()).index(nextPos[1])
                    except:
                        None

            qArr[0][0] = countJ[i] / 6

            actionOut[i] = outDeg
            qVal = 0.0
            # get max V
            for action in actionComb[i]:
                # proximity to another ship
                oldProx = shipProximityQ(ships[i].pos, ships[i].otherPos, Graph, i, 1, nShip,
                                         ships[i].dest, 1)
                prox = shipProximityQ(actionToPos(ships[i].pos, action, Graph)[0], ships[i].otherPos, Graph, i, 1,
                                      nShip, ships[i].dest, 1)
                if oldProx == 1 and prox > 1:
                    prox = 0
                if prox:
                    prox = 1
                if actionToPos(ships[i].pos, action, Graph)[0] == ships[i].dest:
                    prox = 0

                actionJ[i] = action
                countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, action, Graph)[0], ships[i].seen)
                qArrOut = np.zeros(6)
                qArr[0][0] = countJ[i] / 6
                j = 1
                if action != outDeg:
                    # outdegree, Unsensed direction, speed, goal direction, action = stay?, proximity to another ship
                    qArrOut[0] = countJ[i] / 6
                    qArrOut[j] = qClosest[action]
                    qArrOut[j + 2] = 0
                    if goalDir == action:
                        qArrOut[j + 2] = 1
                        qArrOut[0] = 0
                        qArrOut[j] = 0
                    qArrOut[j + 3] = 0
                    qArrOut[j + 4] = prox
                    qArrOut[j+1]= ships[i].maxSpeed
                else:  # if action is stay
                    prox = 0
                    qArrOut[0] = 0
                    qArrOut[j] = 0
                    qArrOut[j + 1] = 0
                    qArrOut[j + 2] = 0
                    qArrOut[j + 3] = 0
                    qArrOut[j + 4] = 0

                newS=actionToPos(state[i], action, Graph)[0]
                arr1=[state[i], state[1-i], newS, ships[i].otherPos[1-i]]+list(qArrOut)
                arrNew=np.array([int(x) for x in arr1])
                arrNew=arrNew.reshape(1, 10)

                if (action == outDeg or qArr[action + 1][0] != 1):

                    qTemp = reconstructed_model_q.predict(arrNew)[0][0] # get action from decision function

                    if qTemp > 0:
                        nearCorner = True
                    if (qTemp > qVal):
                        qVal = qTemp
                        actionOut[i] = action
                        # print(actionOut)
                qArrOut[j + 4] = 0

            actionJ[i] = actionOut[i]

            if actionOut[i] == outDeg:
                ships[i].speed = 0


    return actionOut


def calcNewPos(ships, Graph, t, b, test=1):  # Take action
    global goalSensed
    if test < 0:
        a = 0.6
    state = ()
    actionComb = []
    posOut = []

    for ship in ships:
        state = state + (ship.pos,)
        posOutTemp, actionTemp = ship.getPossibleActions(Graph, ships)
        posOut.append(posOutTemp)
        actionComb.append(actionTemp)

    # make decision
    actionOut = np.full(nShip, outDeg).tolist()
    actionOut = completeDecision(ships, Graph, actionComb, nShip, t)

    # change and update ships and values
    maxTime = 0
    seen = ships[0].seen.copy()
    for i in range(0, nShip):
        # update pos
        ships[i].oldPos = ships[i].pos

        newPos, timeTemp = actionToPos(ships[i].pos, actionOut[i], Graph)
        b[i].append(newPos)

        # update time
        if ships[i].speed != 0:
            timeTemp = timeTemp / ships[i].speed
        ships[i].pos = newPos
        ships[i].time += timeTemp

        # update fuel

        if ships[i].speed!=5:
            print("here", ships[i].speed)
        if ships[i].pos != ships[i].oldPos:
            ships[i].fuel += timeTemp * (0.2525 * math.pow((ships[i].speed + 10) * 2, 2) - 1.6307 * (ships[i].speed + 10) * 2)
        if (timeTemp > maxTime):
            maxTime = timeTemp

        # update seen
        seen.append(ships[i].seen)
        for j in Graph[ships[i].pos].possibleNodes.keys():
            if j not in ships[i].seen:
                ships[i].seen.append(j)
        for k in range(0, nShip):
            if k != i:
                for j in Graph[ships[i].otherPos[k]].possibleNodes.keys():
                    if j not in ships[i].seen:
                        ships[i].seen.append(j)

        # update goal sensed flag
        if ships[i].dest in ships[i].seen:
            goalSensed = True

            for j in range(0, nShip):
                if i != j and ships[j].dest not in ships[j].seen:
                    ships[j].seen.append(ships[j].dest)
                    continue

    # update total time
    for i in range(0, nShip):
        if (ships[i].pos == ships[i].oldPos and ships[i].pos != ships[i].dest):
            ships[i].time += maxTime
    t += 1
    return ships, t, b
ts = 0
te = 0


def mainFunc():
    global goalSensed
    global seen
    global nShip
    global ts,te
    iterTime = 0

    # start time
    ts = time.time()
    # init
    Graph = loadGraph() # load graph
    ships = initShips(Graph)

    locations = {new_list: [ships[new_list].pos] for new_list in range(nShip)}
    crashed = False
    t = 0
    iter = 0

    # seen from initial position
    global seen
    seen = []
    for i in range(0, len(ships)):
        seen.append(ships[i].pos)
        for j in Graph[ships[i].pos].possibleNodes.keys():
            seen.append(j)
    for i in range(0, len(ships)):
        ships[i].seen = seen.copy()

    while not crashed:

        xChange = 0
        refresh = 0

        xChange = 1

        # if end or collision
        if refresh or t > 400 or collisiontDetect(ships):  # if refresh position or collision
            print("COLLIDED")
            ts = time.time()
            goalSensed = False
            seen = []
            exit()

            # refresh positions
            reloadShips(ships, Graph)
            for i in range(0, len(ships)):
                seen.append(ships[i].pos)
                for j in Graph[ships[i].pos].possibleNodes.keys():
                    seen.append(j)
            for i in range(0, len(ships)):
                ships[i].seen = seen.copy()

            t = 0
            iter += 1

        if endDet(ships):  # if ended at destination
            seen = []
            goalSensed = False


            iterTime += printMaxTime(ships)
            print("All at goal")
            print()
            te = time.time()
            print("Code ran for : " + str(te - ts))
            exit()
            time.sleep(5)

            # refresh positions
            reloadShips(ships, Graph)
            for i in range(0, len(ships)):
                seen.append(ships[i].pos)
                for j in Graph[ships[i].pos].possibleNodes.keys():
                    seen.append(j)
            for i in range(0, len(ships)):
                ships[i].seen = seen.copy()
            t = 0
            iter += 1

        if (xChange > 0):
            # take action
            ships, t , locations = calcNewPos(ships, Graph, t, locations, 1)

        if iter == 100:
            break

    quit()


if __name__ == "__main__":

    mainFunc()
