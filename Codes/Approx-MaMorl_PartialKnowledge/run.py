
from random import randint
import numpy as np
import csv
import time
import math
from decision import speedChoice, actionChoice
import pandas as pd
import networkx as nx
import random
import sys
from scipy.spatial import distance



sendInterval = 1 #Ships'sensing raduis
goalSensed = False

"""Set the values here""" 
noOfShip, deg =  [],[] #To run different values so it will run for each instance automatically or just one value for a particular run
noOfShip = [3]  # Set number of ships here
deg = [9] #Set the MAX outdeg here
noOfIter = 10 #Number of iterations for each set of data.
maxSpeed = np.array ([5]*noOfShip[0]) #Set the max speed here, np array size must equal number of ships
filename = "inputs/Varying_Nodes/400nodes_846edges_degree9.csv" #Name of file with input grid
resultPath ='results.csv' # Directory to save the results
noNodes = 400 #number of nodes in the grid
region = 240 #number of nodes in the known region. Keep it 60% of noNodes
cFreq = 3

# Define nodes in rawTest in case using on smaller graph, else leave empty
rawTest = []

def loadGraph1():  # load graph from csv

    with open(filename) as csvfile:
        rawData = list(csv.reader(csvfile, delimiter=','))
    rawData.pop(0)

    retLink = -np.ones((len(rawData), deg[0]), dtype=np.int64)  # Links from a node
    retDist = -np.ones((len(rawData), deg[0]), dtype=np.int64)  # Weights of Links
    retMax = np.full(len(rawData), deg[0] - 1, dtype=np.int64)  # Outdegree of a node
    rawPoints = -np.ones(len(rawData), dtype=np.int64)  # Storing the nodes in raw node ID
    lat_long = {i: [] for i in range(noNodes)} #store nodes lat/long

    n = 0
    k = 0

    for point in rawData:

        retMax[n] = deg[0]
        j = 0
        # Load points from data
        for i in range(0, deg[0]):
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
        for j in range(0, deg[0]):
            if (retLink[i][j] == -1):
                break
            retLink[i][j] = np.where(rawPoints == retLink[i][j])[0]

    return retLink, retDist, retMax, lat_long


# Convert graph format using the NetworkX library
def buildGraph(noNodes, graphLink, graphDist, lat_long):
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

    return G, weights

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


# Find the closet point to boundary region from each ship's location
def findCloset(noOfShip,G, SG, start, ind):

    nearPoint = [-1] * noOfShip[0]
    for i in range(noOfShip[0]):
        start1 = start[i]
        minDist = float("inf")
        for node in list(SG.nodes()):
            dis = distance.euclidean(G.nodes[node]['lat/long'], G.nodes[start1]['lat/long'])
            if dis < minDist:
                minDist = dis
                nearPoint[i] = node
    goalFlag=False
    for i in range(noOfShip[0]):
        if ind in G.neighbors(start[i]):
            goalFlag=True
            break


    if goalFlag:
        startLocs=start[:]
    else:
        startLocs=nearPoint[:]

    return startLocs


# Find shortest path to boundary region using Dijkstra Shortest path algorithm
def shorthestPath(noOfShip, G, nearPoint, start):

    paths = {i: [] for i in range(noOfShip[0])}
    distances = {i: 0 for i in range(noOfShip[0])}
    for i in range(noOfShip[0]):
        paths[i] = nx.dijkstra_path(G, source=start[i], target=nearPoint[i], weight='weight')
        distances[i] = nx.dijkstra_path_length(G, source=start[i], target=nearPoint[i], weight='weight')

    return paths, distances


# Check if there is any intersection between ships' hyper-edge
def Intersect(paths):
    intersect = {i: [] for i in range(len(paths))}

    for k1 in paths:
        for k2 in paths:
            if (k1 != k2) and list(set(paths[k1]) & set(paths[k2])) == []:
                intersect[k1].append(k2)

    return intersect


# Give the order that ships can move without colliding to reach to the region boundary
def shipsOrderAfterGoal(intersect, k):

    shipsOrder=[]
    left=[]
    for i in intersect:

        if i not in left:
            o=[]
            o.append(i)

            for j in intersect:

                if (j not in o) and (j not in left) and all(x in intersect[j] for x in o) and len(o)<k:
                    o.append(j)

            for j in o:

                left.append(j)

            shipsOrder.append(o)

    return shipsOrder

# Calculate partial Fuel to reach boundary
def calObjectiveFuel(locations, weights):
    fuel = [0] * noOfShip[0]
    for i in range(noOfShip[0]):
        for j in range(len(locations[i]) - 1):
            fuel[i] += (weights[locations[i][j], locations[i][j + 1]] / maxSpeed[i]) * (
                        0.2525 * math.pow(maxSpeed[i], 2) + 1.6307 * maxSpeed[i])
    return fuel, sum(fuel)

# Calculate partial Time to reach boundary
def calTimeAfterGoal(distances, order):
    time = 0
    for i in order:

        if len(i) > 1:

            t = []
            for j in i:
                t.append(distances[j] / maxSpeed[j])

            time += max(t)

        else:

            time += distances[i[0]] / maxSpeed[i[0]]

    return time

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

        return actions, tempAction

    def randLoc(self, Graph):  # randomize ship location
        self.pos = randint(0, len(Graph) - 1)  # position
        self.time = 0  # time


    def assignFixedLocs (self,Graph,loc):
        '''Assign a ship location as a fixed parameter, instead of randomly'''
        self.pos = loc  # position
        self.time = 0  # time

    def __init__(self, shipNo, Graph, startLocs, seenNodes, goalNode):  # initialize ship members

        self.Num = shipNo
        self.assignFixedLocs(Graph, loc=startLocs)
        self.dest = goalNode  # destination
        self.pTable = dict()
        self.time = 0
        self.speed = 0 #inital speed of ships
        self.maxSpeed = 5  # max speed of ships
        self.oldPos = self.pos
        self.movPos = [0, 0]
        self.fuel = 0
        self.otherPos = np.full(nShip, 0).tolist()  # position of other ships
        self.seen = seenNodes[:]



def loadGraph():  # get graph from csv
    rawData = []
    global outDeg
    with open(filename) as csvfile:
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




def initShips(Graph, startLocs, seenNodes, goalNode):  # init ship objects

    ships = []
    print("Starting state of n ships: "+ str (startLocs))
    for i in range(0, nShip):
        ship = Ship(i, Graph,startLocs[i], seenNodes, goalNode)
        ships.append(ship)
    ships = np.array(ships)
    return ships

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


def printMaxTime(ships, paths, distances, weights):  # print time at end

    global max1,maxFuel,times,fuels, time1, f
    global shipsOrder

    max1 = 0
    maxFuel = 0

    times = []
    fuels = []
    # calc max values
    for ship in ships:
        if (ship.time > max1):
            max1 = ship.time
        if (ship.fuel > maxFuel):
            maxFuel = ship.fuel
        fuels.append(ship.fuel)
        times.append(ship.time)

    max1 = round(max1, 4)
    maxFuel = round(maxFuel, 2)

    intersect = Intersect(paths)
    shipsOrder = shipsOrderAfterGoal(intersect, nShip)

    fuel, f = calObjectiveFuel(paths, weights)
    time1 = calTimeAfterGoal(distances, shipsOrder)


    print("Max time for n ships is: " + str(max1+time1))
    print("Sum of all fuel is : "+ str(sum(fuels)+f))

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


def completeDecision(ships, Graph, actionComb, nShip, t, qTable):  # calculate actions
    global outDeg
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
                                pArrOut = [count / 6, pArr[action + 1], pClosest[action], 0, 0]
                                if goalDir == action:
                                    pArrOut[4] = 1
                            else:  # if action == 6 / stay
                                pArrOut = [0, 0, 0, 0, 0]
                            pTemp = actionChoice(i, j, action, 0, qTable, ships, None, pArrOut)[1]  # get p value from decision function

                            if pTemp > 0:
                                nearCorner = True
                            if (pTemp > pVal):
                                pVal = pTemp
                                actionJ[j] = action
                                countJ[j] = count
                        if (state[j] == ships[j].dest):
                            actionJ[j] = outDeg
                        pVal = -1000

                        # speed of ship J
                        if actionJ[j] != outDeg:
                            for speed in range(1, ships[j].maxSpeed):
                                pArr[0] = countJ[j] / 6
                                # ParrOut is the follow:
                                # outdegree, proximity to another ship, closest edge, speed, goal direction
                                pArrOut = [count / 6, pArr[actionJ[j] + 1], pClosest[actionJ[j]], speed]
                                pTemp = speedChoice(i, j, actionJ[j], speed, qTable, ships, None, pArrOut)[1]  # get p value from decision function

                                if pTemp > 0:
                                    nearCorner = True
                                if (pTemp > pVal):
                                    pVal = pTemp
                                    speedJ[j] = speed
                        else:
                            speedJ[j] = 0

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
                oldProx = shipProximityQ(ships[i].pos, ships[i].otherPos, Graph, i, 1, nShip, ships[i].dest, 1)
                prox = shipProximityQ(actionToPos(ships[i].pos, action, Graph)[0], ships[i].otherPos, Graph, i, 1, nShip, ships[i].dest, 1)

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

                else:  # if action is stay
                    prox = 0
                    qArrOut[0] = 0
                    qArrOut[j] = 0
                    qArrOut[j + 1] = 0
                    qArrOut[j + 2] = 0
                    qArrOut[j + 3] = 0
                    qArrOut[j + 4] = 0

                if (action == outDeg or qArr[action + 1][0] != 1):

                    qTemp = actionChoice(i, 0, action, 0, qTable, ships, qArrOut, None)[0]  # get action from decision function

                    if qTemp > 0:
                        nearCorner = True
                    if (qTemp > qVal):
                        qVal = qTemp
                        actionOut[i] = action
                qArrOut[j + 4] = 0

            qVal = -1000
            if actionOut[i] != outDeg:
                for speed in range(1, ships[i].maxSpeed):
                    countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, actionOut[i], Graph)[0], ships[i].seen)
                    qArr[0][0] = countJ[i] / 6

                    # outdegree, Unsensed direction, speed, goal direction, action = stay?, proximity to another ship
                    qArrOut[0] = countJ[i] / 6
                    qArrOut[j] = qClosest[actionOut[i]]
                    qArrOut[j + 1] = speed
                    qArrOut[j + 2] = 0
                    if goalDir == actionOut[i]:
                        qArrOut[j + 2] = 1
                    qArrOut[j + 3] = 0
                    qTemp = speedChoice(i, 0, actionOut[i], speed, qTable, ships, qArrOut, None)[0]  # get speed from decision function

                    if (qTemp > qVal):
                        qVal = qTemp
                        ships[i].speed = speed  # update speeds

            else:  # if already at destination
                ships[i].speed = 0

            actionJ[i] = actionOut[i]

            if actionOut[i] == outDeg:
                ships[i].speed = 0

    return actionOut


def calcNewPos(ships, Graph, qTable, t, test=1):  # Take action
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
    actionOut = completeDecision(ships, Graph, actionComb, nShip, t, qTable)

    # change and update ships and values
    maxTime = 0
    seen = ships[0].seen.copy()
    for i in range(0, nShip):
        # update pos
        ships[i].oldPos = ships[i].pos
        newPos, timeTemp = actionToPos(ships[i].pos, actionOut[i], Graph)

        # update time
        if ships[i].speed != 0:
            timeTemp = timeTemp / ships[i].speed
        ships[i].pos = newPos
        ships[i].time += timeTemp

        # update fuel
        if ships[i].pos != ships[i].oldPos:
            ships[i].fuel += timeTemp * (
                        0.2525 * math.pow((ships[i].speed + 10) * 2, 2) - 1.6307 * (ships[i].speed + 10) * 2)
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
    return qTable, ships, t
ts = 0
te = 0


def mainFunc():  # py game run
    global goalSensed
    global seen
    global nShip
    global ts,te
    global maxLen
    global goalNode
    global start


    iterTime = 0
    #Start time
    ts = time.time()
    # init
    graphLink, graphDist, graphMax, lat_long = loadGraph1() #Load graph
    G, weights =buildGraph(noNodes, graphLink, graphDist, lat_long) #convert graph formaat
    SG, maxLen = subGrid(G, region) #Generate connected subgrid as region
    # nodes in the grid that ships sensed them
    seenNodes = list(set(G.nodes) - set(SG.nodes))

    # Randomize ships' starting locations
    start = random.sample(list(set(G.nodes) - set(SG.nodes)), noOfShip[0])
    print("start=", start)

    # randomly select a node in the region as destination
    goalNode = random.choice(list(SG.nodes()))
    print("goalNode=", goalNode)

    # closet node in the boundary of known region
    startLocs=findCloset(noOfShip, G, SG, start, goalNode)
    paths, distances=shorthestPath(noOfShip, G, startLocs, start)

    Graph = loadGraph() # load graph

    # load data
    qTable = np.load("data/qData.npy")
    ships = initShips(Graph, startLocs, seenNodes, goalNode)
    for i in range(0, nShip):
        pTemp = dict()
        for j in range(0, nShip):
            if (i != j):
                pArray = np.load("data/ " + str(i) + '-' + str(j) + '-' + 'pData.npy')
                pTemp.update({j: pArray})
        ships[i].pTable = pTemp
        ships[i].speed = 1


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
            return 1
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


            iterTime += printMaxTime(ships, paths, distances, weights)
            print("All at goal")
            print()
            te = time.time()
            print("Code ran for : " + str(te - ts))
            return 0
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
            qTable, ships, t = calcNewPos(ships, Graph, qTable, t, 1)

        if iter == 100:
            break


    quit()


if __name__ == "__main__":

    # Code for automated running
    for _ in range(noOfIter):
        for i in noOfShip:
            for j in deg:
                nShip = i  # The number of ships here
                outDeg = j # The MAX outdeg here
                failed,count=1,0
                while failed:
                    failed=mainFunc()
                    count += 1 #keep track of number of collisions
                with open(resultPath, mode='a') as csv_file:
                    fieldnames = ['No of Ships', 'out degree', 'Goal Node', 'Subgrid Size','No of Times collided', 'Ship Location', 'order',  'Max time',  'Total Fuel', 'Running time']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    if csv_file.tell() == 0:
                        writer.writeheader()
                    writer.writerow({'No of Ships': nShip, 'out degree': outDeg, 'Goal Node': goalNode, 'Subgrid Size': maxLen,'No of Times collided': count-1, 'Ship Location':str(start),
                      'order': str(shipsOrder),'Max time':str(max1+time1),  'Total Fuel':str(sum(fuels)+f), 'Running time':str(te - ts)})
