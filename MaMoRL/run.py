import pygame
from pygame import gfxdraw
from random import randint
import numpy as np
import itertools as it
import csv
import time
import json
import math
import decision
from matplotlib import pyplot as plt

displayWidth = 1600
displayHeight = 800
sendInterval = 3
goalSensed = False
goalDisplay = True
nShip = 2
seen = []
allStates = []
statesTest = [480, 583, 677, 523, 560, 478, 378, 404, 441, 683, 692, 406, 456, 449, 403, 418, 675, 537, 374, 679, 557,
              576, 377, 439, 545, 390, 702, 376, 384, 388, 577, 543, 519, 556, 455, 397, 587, 578, 446, 450, 467, 375,
              469, 402, 460, 442, 465, 452, 461, 391, 438, 531, 548, 513, 434, 521, 551, 561]

#Define nodes on which to test. See README for more details
rawTest = [24143, 24216, 24218, 24231, 24246, 24530, 24651, 24694, 24696, 24814, 25093, 25107, 25110, 25201, 25361, 26781,
           27088, 27154, 27256, 27311, 27505, 27576, 27611, 27678, 27848, 27870, 27917, 27920, 27977, 27992, 28025, 28214,
           28241, 30748, 31373, 31497, 31730, 33185, 33689, 34770, 34860, 35008, 35053, 35594, 35596, 35657, 35730, 36478,
           36506, 36520, 36807, 37118, 42179, 42262, 42294, 42427, 42813, 43209]


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
        self.drawLoc = [0, 0]
        self.drawLoc[0] = int(
            ((loc[0] - minMax[0]) / float(minMax[2] - minMax[0])) * (displayWidth - 10 - 10) + 10)
        self.drawLoc[1] = int(
            ((loc[1] - minMax[1]) / float(minMax[3] - minMax[1])) * (displayHeight - 10 - 10) + 10)


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
        tempAction = tempAction + (6,)
        actions = actions + (ships[self.Num].pos,)
        return actions, tempAction

    def randLoc(self, Graph, iter):  # randomize ship location
        self.pos = randint(0, len(statesTest) - 1)
        self.pos = allStates[iter][self.Num]
        if self.Num == 0:
            self.pos = 6
        if self.Num == 1:
            self.pos = 37
        self.time = 0
        self.oldPos = self.pos
        self.movPos = [Graph[self.oldPos].drawLoc[0], Graph[self.oldPos].drawLoc[1]]

    def __init__(self, shipNo, Graph):  # initialize ship members
        self.Num = shipNo
        self.randLoc(Graph, 0)
        self.dest = 0 #377
        self.pTable = dict()
        self.time = 0
        self.speed = 1
        self.maxSpeed = 5
        self.oldPos = self.pos
        self.movPos = [0, 0]
        self.fuel = 0
        self.otherPos = 0
        self.seen = []
        self.movPos = [Graph[self.oldPos].drawLoc[0], Graph[self.oldPos].drawLoc[1]]


def loadGraph():  # get graph from csv
    rawData = []
    with open('inputMap.csv') as csvfile: #Set name of input grid here
        rawData = list(csv.reader(csvfile, delimiter=','))
    rawData.pop(0)
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
        for i in range(0, 6):
            if (point[i * 2 + 3] == 'N/A'):
                break
            if int(point[i * 2 + 3]) in rawTest:
                nodePossible.update({int(point[i * 2 + 3]): int(point[i * 2 + 1 + 3])})
        if(int(point[0]) in rawTest):
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
            # Test
            # if outPoint in statesTest and i in statesTest:
            #     newPossible.update({outPoint: Graph[i].possibleNodes[connPoint]})
            newPossible.update({outPoint: Graph[i].possibleNodes[connPoint]})

        Graph[i].possibleNodes = newPossible
    return Graph


def djikstra(pos, Graph, shipSeen):  # djikstra algorithm
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


def drawGraph(Graph, ships, gameDisplay):  # draw nodes and lines
    circleColorInner = (41, 128, 185)
    circleColorOuter = (189, 195, 199)
    circleNotSeenColor = (69, 69, 69)
    lineColorInner = (218, 79, 73)
    fontColor = (65, 65, 65)
    circleSize = 5
    fontSize = 12

    font = pygame.font.SysFont("timesnewroman", fontSize)

    for point in Graph:
        nodePos = point.drawLoc
        circleColorOuter = (189, 195, 199)

        for connPoint in Graph:
            if (connPoint.pos in point.possibleNodes.keys()):
                connPos = connPoint.drawLoc

                # draw line distances
                textPos = [(connPos[0] + nodePos[0]) / 2,
                           (connPos[1] + nodePos[1]) / 2]

                # draw edges
                pygame.draw.aaline(gameDisplay, fontColor, nodePos, connPos)
                # gfxdraw.filled_circle(gameDisplay, int((connPos[0] - nodePos[0])* 0.9 + nodePos[0]), int((connPos[1] - nodePos[1])* 0.9 + nodePos[1]), 2, fontColor)
                text = font.render(str(point.possibleNodes[connPoint.pos]), True, fontColor)
                gameDisplay.blit(text, textPos)

                # draw nodes
                circleColorInner = (41, 128, 185)
                if connPoint.pos not in ships[0].seen and connPoint.pos not in ships[1].seen:
                    circleColorInner = (69, 69, 69)
                gfxdraw.filled_circle(gameDisplay, connPos[0], connPos[1], circleSize, circleColorOuter)
                for k in range(-1, 3):
                    gfxdraw.aacircle(gameDisplay, connPos[0], connPos[1], circleSize + k, circleColorInner)

                text = font.render(str(connPoint.pos), True, fontColor)
                gameDisplay.blit(text, connPos)

        circleColorInner = (41, 128, 185)
        # print(point.pos)
        if point.pos not in ships[0].seen and point.pos not in ships[1].seen:
            circleColorInner = (69, 69, 69)
        gfxdraw.filled_circle(gameDisplay, nodePos[0], nodePos[1], circleSize, circleColorOuter)
        for k in range(-1, 3):
            gfxdraw.aacircle(gameDisplay, nodePos[0], nodePos[1], circleSize + k, circleColorInner)

        text = font.render(str(point.pos), True, fontColor)
        gameDisplay.blit(text, nodePos)


def drawShips(Graph, ships, gameDisplay, t):  # draw objects related to ship in graph
    global goalDisplay
    circleColorInner = (218, 79, 73)
    circleSize = 5
    fontDestColor = (218, 79, 73)

    shipImg = pygame.image.load('ship2.png')
    waitImage = pygame.image.load('wait.png')
    sendImage = pygame.image.load('send.png')
    font = pygame.font.SysFont("timesnewromanbold", 14)
    fontSpeed = pygame.font.SysFont("timesnewromanbold", 18)

    for shipI in ships:
        oldPos = [Graph[shipI.oldPos].drawLoc[0], Graph[shipI.oldPos].drawLoc[1]]
        newPos = [Graph[shipI.pos].drawLoc[0], Graph[shipI.pos].drawLoc[1]]
        ships[shipI.Num].movPos[1] = int(ships[shipI.Num].movPos[1] + (newPos[1] - oldPos[1]) / (15 - shipI.speed))
        ships[shipI.Num].movPos[0] = int(ships[shipI.Num].movPos[0] + (newPos[0] - oldPos[0]) / (15 - shipI.speed))
        if ((oldPos[0] > newPos[0] and ships[shipI.Num].movPos[0] < newPos[0]) or (
                oldPos[0] < newPos[0] and ships[shipI.Num].movPos[0] > newPos[0])):
            ships[shipI.Num].movPos[0] = newPos[0]
        if ((oldPos[1] > newPos[1] and ships[shipI.Num].movPos[1] < newPos[1]) or (
                oldPos[1] < newPos[1] and ships[shipI.Num].movPos[1] > newPos[1])):
            ships[shipI.Num].movPos[1] = newPos[1]

        nodePos = [ships[shipI.Num].movPos[0] - 18, ships[shipI.Num].movPos[1] - 10]
        destPos = Graph[shipI.dest].drawLoc

        # draw goal
        if (shipI.Num == 0):
            for k in range(-1, 3):
                gfxdraw.aacircle(gameDisplay, destPos[0], destPos[1], circleSize + k, circleColorInner)

        # draw ship number
        gameDisplay.blit(shipImg, nodePos)
        text = font.render(str(shipI.Num), True, (65, 65, 65))
        gameDisplay.blit(text, (nodePos[0] + 32, nodePos[1] + 12))

        # draw ship image
        if (shipI.oldPos == shipI.pos and shipI.dest != shipI.pos):
            gameDisplay.blit(waitImage, (nodePos[0] + 37, nodePos[1] - 5))

        if (t % sendInterval == 0 and shipI.dest != shipI.pos) or (goalSensed and goalDisplay):
            goalDisplay = False
            gameDisplay.blit(sendImage, (nodePos[0] + 37, nodePos[1] - 5))

        # draw speed
        text = fontSpeed.render("SPEED: " + str(shipI.speed), True, fontDestColor)
        gameDisplay.blit(text, (nodePos[0] - 3, nodePos[1] + 25))
        text = fontSpeed.render("(MAX: " + str(shipI.maxSpeed) + ")", True,
                                fontDestColor)
        gameDisplay.blit(text, (nodePos[0] - 1, nodePos[1] + 25 + 15))


def initShips(Graph):  # init ship objects
    ships = []
    pTemp = (np.full((len(Graph), len(Graph), 7, 6), (1 / (7))))  # initialize p table
    for i in range(0, nShip):
        ship = Ship(i, Graph)
        pTable = dict()
        for j in range(0, nShip):
            if (i != j):
                pTable.update({j: pTemp})
        ship.pTable = pTable
        ships.append(ship)
    ships = np.array(ships)
    ships[0].speed = 1
    ships[1].speed = 1
    ships[0].maxSpeed = 5
    ships[1].maxSpeed = 10
    return ships


def detAdjNum(Graph, n, shipSeen):
    count = 0
    for i in Graph[n].possibleNodes.keys():
        # print(i, n)
        if i in shipSeen:
            count += 1
    # print(count)
    return count


def initQTable(Graph):  # init Q variable
    qTable = []
    qTable = np.full((((len(Graph),) * nShip) + ((7,) * nShip)), (1 / (7 ** nShip)))
    return qTable


def reloadShips(ships, Graph, iter):  # randomize ships
    for i in range(0, nShip):
        ships[i].randLoc(Graph, iter)
    return ships


def collisiontDetect(ships):  # if collision occurs
    for i in range(0, nShip):
        for j in range(i, nShip):
            if (i != j and ships[i].pos == ships[j].pos and ships[i].pos != ships[i].dest and ships[j].pos != ships[
                j].dest):
                return True
    return False


def endDet(ships):  # if all ships reach goal
    for ship in ships:
        if (ship.pos != ship.dest):
            return False
    return True


def endDetPic(ships, Graph):  # if all ships reach goal
    for ship in ships:
        if (ship.movPos != Graph[ship.dest].drawLoc):
            return False
    return True


def printMaxTime(ships, gameDisplay):  # print time at end
    fontColor = (65, 65, 65)
    bgColor = (236, 240, 241)
    lineColorInner = (218, 79, 73)
    max = 0
    maxFuel = 0
    for ship in ships:
        if (ship.time > max):
            max = ship.time
        if (ship.fuel > maxFuel):
            maxFuel = ship.fuel
    max = round(max, 4)
    maxFuel = round(maxFuel, 2)
    pygame.draw.rect(gameDisplay, bgColor, (650, 350, 300, 100))
    pygame.draw.rect(gameDisplay, lineColorInner, (650, 350, 300, 100), 5)
    font = pygame.font.SysFont("timesnewroman", 24)
    text = font.render("Time Taken: " + str(max) + " units", True, fontColor)
    gameDisplay.blit(text, (800 - 120, 400 - 40))

    font = pygame.font.SysFont("timesnewroman", 24)
    text = font.render("Max Fuel: " + str(maxFuel) + " units", True, fontColor)
    gameDisplay.blit(text, (800 - 120, 400))
    return max



def getPossible(ships, pos, i, Graph):  # possible actions from node
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
    if (action == 6):
        return pos, 0
    # print("ACtION", Graph[pos].possibleNodes.values(), action)
    return list(Graph[pos].possibleNodes.keys())[action], list(Graph[pos].possibleNodes.values())[action]


def calcNewPos(ships, Graph, qTable, t, test=1):  # Take action
    global goalSensed
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
    actionOut = [6, 6]
    for i in range(0, nShip):
        if (ships[i].pos != ships[i].dest):
            state = [0, 0]
            state[i] = ships[i].pos
            for j in range(0, nShip):
                if i != j:
                    if t % 3 == 0:
                        state[j] = ships[j].pos
                        ships[i].otherPos = ships[j].pos
                    state[j] = ships[i].otherPos
                    otherComb = np.arange(len(Graph[ships[i].otherPos].possibleNodes.keys()) - 1).tolist()
                    otherComb.append(6)
            actionJ = [6, 6]
            countJ = [0, 0]
            nearCorner = False
            # get max P
            for j in range(0, nShip):
                if (i != j):
                    pVal = -1000
                    for action in otherComb:
                        # print(ships[i].pTable[j][ships[0].pos][ships[1].pos][action][0])
                        # print("CRASH", action, otherComb, actionComb[j])
                        count = detAdjNum(Graph, actionToPos(state[j], action, Graph)[0], ships[j].seen)
                        pTemp = ships[i].pTable[j][ships[0].pos][state[j] ][action][count]
                        if pTemp > 0:
                            nearCorner = True
                        if (pTemp > pVal):
                            pVal = pTemp
                            actionJ[j] = action
                            countJ[j] = count
                    if (state[j]  == ships[j].dest):
                        actionJ[j] = 6

                    if goalSensed:
                        _, dist, prev = djikstra(state[j] , Graph, ships[j].seen)
                        nextPos = prev[ships[j].dest]
                        nextAction = 6
                        try:
                            nextAction = list(Graph[state[j] ].possibleNodes.keys()).index(nextPos[1])
                        except:
                            None
                        countJ[j] = detAdjNum(Graph, actionToPos(state[j] , nextAction, Graph)[0], ships[j].seen)
                        actionJ[j] = nextAction

                    if (nearCorner == False) and state[j]  != ships[j].dest  and (not goalSensed or dist[ships[j].dest] == 1000000):
                        corners, _, prev = djikstra(state[j] , Graph, ships[j].seen)
                        nextPos = prev[min(corners, key=corners.get)]
                        nextAction = 6
                        # print(list(Graph[ships[j].pos].possibleNodes.keys()), nextPos)
                        try:
                            nextAction = list(Graph[state[j] ].possibleNodes.keys()).index(nextPos[1])
                        except:
                            None
                        countJ[j] = detAdjNum(Graph, actionToPos(state[j] , nextAction, Graph)[0], ships[j].seen)
                        actionJ[j] = nextAction
                        # print(i, j, corners, nextAction)
                    # print("P Out", nearCorner, goalSensed, actionJ[j], otherComb)
                    ships[i].otherPos, _ = actionToPos(ships[i].otherPos, actionJ[j], Graph)
            qVal = -10000
            nearCorner = False
            actionJ[i] = 0
            countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, 0, Graph)[0], ships[i].seen)
            qMat = tuple(state) + tuple(actionJ) + tuple(countJ)
            base = qTable[qMat]
            if base < 0:
                actionJ[i] = 1
                countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, 1, Graph)[0], ships[i].seen)
                qMat = tuple(state) + tuple(actionJ) + tuple(countJ)
                base = qTable[qMat]
            # get max V
            for action in actionComb[i]:
                actionJ[i] = action
                countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, action, Graph)[0], ships[i].seen)
                qMat = tuple(state) + tuple(actionJ) + tuple(countJ)
                qTemp = qTable[qMat]
                # print("Q", qTemp, base, action, actionToPos(ships[i].pos, action, Graph)[0], countJ, nearCorner)
                if qTemp > 0 and qTemp != base:
                    nearCorner = True
                if (qTemp > qVal):
                    qVal = qTemp
                    actionOut[i] = action

            # print("Q Out", actionOut[i], state, actionJ, countJ)
            if goalSensed:
                _, dist, prev = djikstra(ships[i].pos, Graph, ships[i].seen)
                # print("GOAL FOUND", dist[ships[i].dest], prev[ships[i].dest], ships[i].seen)
                if(dist[ships[i].dest] != 1000000):
                    nextPos = prev[ships[i].dest]
                    nextAction = 6
                    try:
                        nextAction = list(Graph[ships[i].pos].possibleNodes.keys()).index(nextPos[1])
                    except:
                        None
                    countJ[i] = detAdjNum(Graph, actionToPos(ships[i].pos, nextAction, Graph)[0], ships[i].seen)
                    actionJ[i] = nextAction
                    actionOut[i] = nextAction

            if (nearCorner == False) and (not goalSensed or dist[ships[i].dest] == 1000000):
                # print("NEXT CORNER")
                corners, _, prev = djikstra(ships[i].pos, Graph, ships[i].seen)
                crash = -1
                while crash == -1:
                    nextAction = 6
                    # print(list(Graph[ships[j].pos].possibleNodes.keys()), nextPos)
                    # print(nextPos, corners, minOpen)
                    try:
                        minOpen = min(corners, key=corners.get)
                        nextPos = prev[minOpen]
                        nextAction = list(Graph[ships[i].pos].possibleNodes.keys()).index(nextPos[1])
                    except:
                        None
                        # print("ERROR", Graph[ships[i].pos].possibleNodes.keys(), nextPos, detAdjNum(Graph, minOpen, ships[i].seen), len(Graph[minOpen].possibleNodes.keys()))
                    countJ[j] = detAdjNum(Graph, actionToPos(ships[i].pos, nextAction, Graph)[0], ships[i].seen)

                    actionOut[i] = nextAction
                    actionJ[i] = nextAction
                    qMat = tuple(state) + tuple(actionJ) + tuple(countJ)
                    crash = qTable[qMat]
                    try:
                        corners.pop(minOpen)
                    except:
                        None
                    # print(i, j, corners, nextAction)

            actionJ[i] = actionOut[i]
            speedJ = [1, 1]

            # speed action from module
            # decision.speedDecisionTest(qTable, ships, state, actionJ, speedJ, i)

            # print("Q Out", actionOut[i], state, actionJ, countJ)
            if actionOut[i] == 6:
                ships[i].speed = 0

    maxTime = 0
    for i in range(0, nShip):
        # print("OUT", ships[i].pos, actionOut[i])
        ships[i].oldPos = ships[i].pos
        ships[i].movPos = [Graph[ships[i].oldPos].drawLoc[0], Graph[ships[i].oldPos].drawLoc[1]]
        newPos, timeTemp = actionToPos(ships[i].pos, actionOut[i], Graph)
        if ships[i].speed != 0:
            timeTemp = timeTemp / ships[i].speed
        ships[i].pos = newPos
        ships[i].time += timeTemp
        if ships[i].pos != ships[i].oldPos:
            ships[i].fuel += timeTemp * (0.2525 * math.pow((ships[i].speed + 10) * 2, 2) - 1.6307 * (ships[i].speed + 10) * 2)
            # print("GG", ships[i].fuel, timeTemp, ships[i].speed)
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
        # print(goalSensed)
        # print(seen, ships[i].seen)
    for i in range(0, nShip):
        if (ships[i].pos == ships[i].oldPos and ships[i].pos != ships[i].dest):
            ships[i].time += maxTime

    t += 1
    return qTable, ships, t


def initPyGame():  # py game run
    global goalSensed
    global seen
    global allStates
    # init
    Graph = loadGraph()
    statesK = []
    for i in range(0, len(Graph)):
        for j in range(0, len(Graph)):
            if i != j:
                statesK.append([i, j])
    allStates = statesK
    ships = initShips(Graph)
    qTable = initQTable(Graph)
    # load data
    try:
        qTable = np.load("qData.npy")
        for i in range(0, 2):
            pTemp = dict()
            for j in range(0, 2):
                if (i != j):
                    pArray = np.load(str(i) + '-' + str(j) + '-' + 'pData.npy')
                    pTemp.update({j: pArray})
            ships[i].pTable = pTemp
        qTable = np.array(qTable)
    except:
        print("NO DATA")

    pygame.init()
    bgColor = (236, 240, 241)
    gameDisplay = pygame.display.set_mode((displayWidth, displayHeight))
    pygame.display.set_caption("ONR Ship Trial - UNDIRECTED")
    clock = pygame.time.Clock()
    crashed = False
    t = 0
    iter = 0
    iterTime = []
    show = [0, 0, 0]
    global seen
    seen = []
    # ships[0].pos = 446
    # ships[1].pos = 388
    for i in range(0, len(ships)):
        seen.append(ships[i].pos)
        for j in Graph[ships[i].pos].possibleNodes.keys():
            seen.append(j)
    for i in range(0, len(ships)):
        ships[i].seen = seen.copy()
    iterTime = []
    while not crashed:
        xChange = 0
        refresh = 0
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                crashed = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    xChange = -1
                if event.key == pygame.K_RIGHT:
                    xChange = 1
                if event.key == pygame.K_SPACE:
                    refresh = 1
                if event.key == pygame.K_ESCAPE:
                    exit()
                if event.key == pygame.K_1:
                    if (show[0] == 1):
                        show[0] = 0
                    else:
                        show[0] = 1
                if event.key == pygame.K_2:
                    if (show[1] == 1):
                        show[1] = 0
                    else:
                        show[1] = 1
                if event.key == pygame.K_3:
                    if (show[2] == 1):
                        show[2] = 0
                    else:
                        show[2] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT or event.key == pygame.K_LEFT or event.key == pygame.K_SPACE:
                    xChange = 0
                    refresh = 0

        # if end or collision
        if refresh or t > 50 or collisiontDetect(ships):
            goalSensed = False
            seen = []
            gameDisplay.fill(bgColor)
            drawGraph(Graph, ships, gameDisplay)
            drawShips(Graph, ships, gameDisplay, t)
            pygame.display.update()
            reloadShips(ships, Graph, 0)
            for i in range(0, len(ships)):
                seen.append(ships[i].pos)
                for j in Graph[ships[i].pos].possibleNodes.keys():
                    seen.append(j)
            for i in range(0, len(ships)):
                ships[i].seen = seen.copy()
            t = 0
            iter += 1
        if endDetPic(ships, Graph):
            seen = []
            goalSensed = False
            gameDisplay.fill(bgColor)
            drawGraph(Graph, ships, gameDisplay)
            drawShips(Graph, ships, gameDisplay, t)
            iterTime.append([iter, printMaxTime(ships, gameDisplay)])
            pygame.display.update()
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
            # take action
            qTable, ships, t = calcNewPos(ships, Graph, qTable, t, 1)

        gameDisplay.fill(bgColor)
        drawGraph(Graph, ships, gameDisplay)
        drawShips(Graph, ships, gameDisplay, t)
        pygame.display.update()
        clock.tick(120)
        if iter == 3305:
            break

    print(iterTime)
    plt.plot(iterTime)
    plt.show()


if __name__ == "__main__":
    n = 3
    m = 10
    maxDist = 5
    noOfShips = 3
    initPyGame()
