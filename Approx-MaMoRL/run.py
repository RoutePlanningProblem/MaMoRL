import pygame
from pygame import gfxdraw
from random import randint
import numpy as np
import csv
import time
import json
import math
from decision import speedChoice, actionChoice

# change
displayWidth = 1400
displayHeight = 770
sendInterval = 1
goalSensed = False
goalDisplay = True


"""Set the values here"""
nShip = 6  # Set number of ships here
outDeg = 9 #Set the MAX outdeg here
goalNode = 0 # Set the node index you want as the goal
"""Ensure you set the input grid below on line 119"""

rawTest = []
# Define nodes in rawTest in case using on smaller graph


"""
Note some parts of code have UI related code, which we do not utilize 

"""

class node:

    def getPosition(self):  # get adjacant nodes
        return self.pos

    def addPossible(self, addDict):  # add adjacant nodes
        self.possibleNodes.update(addDict)

    def getDistance(self, node):  # get distance to adjascant node
        return self.possibleNodes[node]

    def setDrawLoc(self, loc, minMax):
        self.drawLoc[0] = int(
            ((loc[0] - minMax[0]) / float(minMax[2] - minMax[0])) * (displayWidth - 60 - 60) + 60)
        self.drawLoc[1] = int(
            ((loc[1] - minMax[1]) / float(minMax[3] - minMax[1])) * (displayHeight - 60 - 60) + 60)

    def __init__(self, addPos, loc, index):  # init node members
        self.possibleNodes = dict()  # links
        self.pos = index  # position
        self.rawPos = addPos
        self.loc = loc
        self.drawLoc = [0, 0]  # draw location


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
        self.movPos = [Graph[self.oldPos].drawLoc[0], Graph[self.oldPos].drawLoc[1]]  # for drawing movement


    def assignFixedLocs (self,Graph,loc):
        '''Assign a ship location as a fixed parameter, instead of randomly'''
        self.pos = loc  # position
        self.time = 0  # time
        self.oldPos = self.pos  # for drawing movement
        self.movPos = [Graph[self.oldPos].drawLoc[0], Graph[self.oldPos].drawLoc[1]]  # for drawing movement

    def __init__(self, shipNo, Graph):  # initialize ship members
        global goalNode
        self.Num = shipNo
        self.randLoc(Graph) # randomly reloc ships
        #self.assignFixedLocs(Graph, loc=startLoc)
        self.dest = goalNode  # dest
        self.pTable = dict()
        self.time = 0
        self.speed = 0
        self.maxSpeed = 5  # change
        self.oldPos = self.pos
        self.movPos = [0, 0]
        self.fuel = 0
        self.otherPos = np.full(nShip, 0).tolist()  # pos of other ships
        self.seen = []
        self.movPos = [Graph[self.oldPos].drawLoc[0], Graph[self.oldPos].drawLoc[1]]


def loadGraph():  # get graph from csv
    rawData = []
    global outDeg
    with open('input\\400nodes_846edges_degree9.csv') as csvfile:  #Place grid name here
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

    for points in rawData:
        if (float(points[1]) not in xPoints):
            xPoints.append(float(points[1]))
        if (float(points[2]) not in yPoints):
            yPoints.append(float(points[2]))

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
        Graph[i].setDrawLoc(Graph[i].loc, minMax)
    return Graph


# also stores points towards the node
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


def drawGraph(Graph, ships, gameDisplay):  # draw nodes and lines
    bgColor = np.array([236, 240, 241])
    fontColor = np.array([65, 65, 65])
    nodeFontColor = np.array([0, 21, 64])
    circleSize = 10
    edgeWidth = 1

    font = pygame.font.SysFont("timesnewroman", 12)
    nodeFont = pygame.font.SysFont("timesnewroman", 14)

    centerFont = -5

    for point in Graph:
        nodePos = point.drawLoc
        circleColorOuter = np.array([189, 195, 199])

        for connPoint in Graph:
            if (connPoint.pos in point.possibleNodes.keys()):
                connPos = connPoint.drawLoc

                seenConn = False
                seenPoint = False

                # draw nodes
                circleColorInner = (np.array([69, 69, 69]) + bgColor * 2) / 3
                for ship in range(0, nShip):
                    if point.pos in ships[ship].seen:
                        seenPoint = True
                    if connPoint.pos in ships[ship].seen:
                        circleColorInner = np.array([0, 49, 110])
                        seenConn = True

                # draw line distances
                textPos = [(connPos[0] + nodePos[0]) / 2,
                           (connPos[1] + nodePos[1]) / 2]

                # draw edges
                if (nodePos[0] - connPos[0]) != 0:
                    angle = math.atan((nodePos[1] - connPos[1]) / (nodePos[0] - connPos[0]))
                else:
                    angle = 90
                yDir = -1
                if connPos[1] > nodePos[1]:
                    yDir = 1
                xDir = -1
                if connPos[0] > nodePos[0]:
                    xDir = 1
                drawLoc = (
                (connPos[0] - math.sin(angle) * edgeWidth * xDir, connPos[1] - math.cos(angle) * edgeWidth * yDir),
                (connPos[0] + math.sin(angle) * edgeWidth * xDir, connPos[1] + math.cos(angle) * edgeWidth * yDir),
                (nodePos[0] + math.sin(angle) * edgeWidth * xDir, nodePos[1] + math.cos(angle) * edgeWidth * yDir),
                (nodePos[0] - math.sin(angle) * edgeWidth * xDir, nodePos[1] - math.cos(angle) * edgeWidth * yDir))

                # draw only the grayed line if not seen
                if not seenConn or not seenPoint:
                    pygame.gfxdraw.filled_polygon(gameDisplay, drawLoc, (fontColor + bgColor * 2) / 3)
                    pygame.gfxdraw.aapolygon(gameDisplay, drawLoc, (fontColor + bgColor * 2) / 3)

                # Draw node if it is see
                else:
                    pygame.gfxdraw.filled_polygon(gameDisplay, drawLoc, fontColor)
                    pygame.gfxdraw.aapolygon(gameDisplay, drawLoc, fontColor)
                    text = font.render(str(point.possibleNodes[connPoint.pos]), True, fontColor)
                    gameDisplay.blit(text, textPos)

                gfxdraw.filled_circle(gameDisplay, connPos[0], connPos[1], circleSize + 2, circleColorInner)
                gfxdraw.filled_circle(gameDisplay, connPos[0], connPos[1], circleSize, circleColorOuter)
                for k in range(-1, 3):
                    gfxdraw.aacircle(gameDisplay, connPos[0], connPos[1], circleSize + k, circleColorInner)

                # Draw edge weight if seen
                if seenConn:
                    text = nodeFont.render(str(connPoint.pos), True, bgColor)
                    wNode = text.get_rect().width
                    hNode = text.get_rect().height
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            gameDisplay.blit(text,
                                             (connPos[0] - wNode / 2 + 1 / 2 + i, connPos[1] - hNode / 2 + 1 / 2 + j))
                    text = nodeFont.render(str(connPoint.pos), True, nodeFontColor)
                    wNode = text.get_rect().width
                    hNode = text.get_rect().height
                    gameDisplay.blit(text, (connPos[0] - wNode / 2 + 1 / 2, connPos[1] - hNode / 2 + 1 / 2))

        # draw nodes again after edges
        circleColorInner = (np.array([69, 69, 69]) + bgColor * 2) / 3
        for ship in range(0, 1):
            if point.pos in ships[ship].seen:
                # change circle color if seen
                circleColorInner = np.array([0, 49, 110])

        gfxdraw.filled_circle(gameDisplay, nodePos[0], nodePos[1], circleSize + 2, circleColorInner)
        gfxdraw.filled_circle(gameDisplay, nodePos[0], nodePos[1], circleSize, circleColorOuter)
        for k in range(-1, 3):
            gfxdraw.aacircle(gameDisplay, nodePos[0], nodePos[1], circleSize + k, circleColorInner)

        # Draw node number if seen
        if seenPoint:
            text = nodeFont.render(str(point.pos), True, bgColor)
            wNode = text.get_rect().width
            hNode = text.get_rect().height
            for i in range(-1, 2):
                for j in range(-1, 2):
                    gameDisplay.blit(text, (nodePos[0] - wNode / 2 + 1 / 2 + i, nodePos[1] - hNode / 2 + 1 / 2 + j))
            text = nodeFont.render(str(point.pos), True, nodeFontColor)
            wNode = text.get_rect().width
            hNode = text.get_rect().height
            gameDisplay.blit(text, (nodePos[0] - wNode / 2 + 1 / 2, nodePos[1] - hNode / 2 + 1 / 2))


def drawShips(Graph, ships, gameDisplay, t):  # draw objects related to ship in graph
    global goalDisplay
    circleColorInner = (218, 79, 73)
    bgColor = np.array([236, 240, 241])
    circleSize = 10
    speedColor = (0, 21, 64)
    radiusColor = (0, 140, 80)

    shipImg = pygame.image.load('resources/ship3.png')
    waitImage = pygame.image.load('resources/wait.png')
    sendImage = pygame.image.load('resources/send.png')
    font = pygame.font.SysFont("timesnewromanbold", 24)
    fontSpeed = pygame.font.SysFont("timesnewromanbold", 16)

    for shipI in ships:
        oldPos = [Graph[shipI.oldPos].drawLoc[0], Graph[shipI.oldPos].drawLoc[1]]
        newPos = [Graph[shipI.pos].drawLoc[0], Graph[shipI.pos].drawLoc[1]]
        ships[shipI.Num].movPos[1] = int(ships[shipI.Num].movPos[1] + (newPos[1] - oldPos[1]) / (15 - shipI.speed))
        ships[shipI.Num].movPos[0] = int(ships[shipI.Num].movPos[0] + (newPos[0] - oldPos[0]) / (15 - shipI.speed))

        # calculate movpos - pos when ship is moving
        if ((oldPos[0] > newPos[0] and ships[shipI.Num].movPos[0] < newPos[0]) or (
                oldPos[0] < newPos[0] and ships[shipI.Num].movPos[0] > newPos[0])):
            ships[shipI.Num].movPos[0] = newPos[0]
        if ((oldPos[1] > newPos[1] and ships[shipI.Num].movPos[1] < newPos[1]) or (
                oldPos[1] < newPos[1] and ships[shipI.Num].movPos[1] > newPos[1])):
            ships[shipI.Num].movPos[1] = newPos[1]

        nodePos = [ships[shipI.Num].movPos[0] - 18, ships[shipI.Num].movPos[1] - 10]
        destPos = Graph[shipI.dest].drawLoc

        # Draw sensing radius
        for point in Graph[shipI.pos].possibleNodes.keys():
            shipOnPoint = False
            for j in range(0, nShip):
                if ships[j].pos == point:
                    shipOnPoint = True
            if not shipOnPoint:
                for k in range(-1, 3):
                    gfxdraw.aacircle(gameDisplay, Graph[point].drawLoc[0], Graph[point].drawLoc[1], circleSize + 8 + k,
                                     radiusColor)

        # draw goal
        for j in range(0, nShip):
            if ships[j].pos == point or j == shipI.Num:
                for k in range(-1, 3):
                    gfxdraw.aacircle(gameDisplay, destPos[0], destPos[1], circleSize + k, circleColorInner)
                for k in range(-1, 3):
                    gfxdraw.aacircle(gameDisplay, destPos[0], destPos[1], circleSize + k + 8, circleColorInner)

        # draw ship number
        gameDisplay.blit(shipImg, nodePos)
        for i in range(-1, 2):
            for j in range(-1, 2):
                text = font.render(str(shipI.Num), True, bgColor)
                gameDisplay.blit(text, (nodePos[0] + 32 + i, nodePos[1] + 12 + j))
        text = font.render(str(shipI.Num), True, (65, 65, 65))
        gameDisplay.blit(text, (nodePos[0] + 32, nodePos[1] + 12))

        # draw ship image
        shipXdisplace = 42
        shipYdisplace = -15
        if (shipI.oldPos == shipI.pos and shipI.dest != shipI.pos):
            gameDisplay.blit(waitImage, (nodePos[0] + shipXdisplace, nodePos[1] + shipYdisplace))

        # draw wifi signal
        if (t % sendInterval == 0 and shipI.dest != shipI.pos) or (goalSensed and goalDisplay):
            goalDisplay = False
            gameDisplay.blit(sendImage, (nodePos[0] + shipXdisplace, nodePos[1] + shipYdisplace))

        # draw speed
        speedXDisplace = -3
        speedYDisplace = 30
        maxYDisplace = 15
        maxXDisplace = - 1
        for i in range(-1, 2):
            for j in range(-1, 2):
                text = fontSpeed.render("SPEED: " + str(shipI.speed), True, bgColor)
                gameDisplay.blit(text, (nodePos[0] + speedXDisplace + i, nodePos[1] + speedYDisplace + j))
                text = fontSpeed.render("(MAX: " + str(shipI.maxSpeed) + ")", True,
                                        bgColor)
                gameDisplay.blit(text, (nodePos[0] + maxXDisplace + i, nodePos[1] + speedYDisplace + maxYDisplace + j))
        text = fontSpeed.render("SPEED: " + str(shipI.speed), True, speedColor)
        gameDisplay.blit(text, (nodePos[0] + speedXDisplace, nodePos[1] + speedYDisplace))
        text = fontSpeed.render("(MAX: " + str(shipI.maxSpeed) + ")", True,
                                speedColor)
        gameDisplay.blit(text, (nodePos[0] + maxXDisplace, nodePos[1] + speedYDisplace + maxYDisplace))

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

"""
def initShips(Graph):  # init ship objects
    global startLocs  # Starting location of n ships, defined at top
    ships = []
    print("Starting state of n ships: "+ str (startLocs))
    for i in range(0, nShip):
        ship = Ship(i, Graph,startLocs[i])
        ships.append(ship)
    ships = np.array(ships)
    return ships
"""
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


def endDetPic(ships, Graph):  # if all ships reach goal
    for ship in ships:
        if (ship.movPos != Graph[ship.dest].drawLoc):
            return False
    return True


def printMaxTime(ships):  # print time at end
    fontColor = (65, 65, 65)
    bgColor = (236, 240, 241)
    lineColorInner = (218, 79, 73)
    max = 0
    maxFuel = 0

    # calc max values
    times = []
    fuels = []
    # calc max values
    for ship in ships:
        if (ship.time > max):
            max = ship.time
        if (ship.fuel > maxFuel):
            maxFuel = ship.fuel
        fuels.append(ship.fuel)
        times.append(ship.time)
    max = round(max, 4)
    maxFuel = round(maxFuel, 2)

    print("Times for n ships : " + str(times))
    print("Fuels for n ships : " + str(fuels))
    print()

    print("Avg time for n ships is : " + str(sum(times) / len(times)))
    print("Avg Fuel for n ships is : " + str(sum(fuels) / len(fuels)))
    print()

    print("Max time for n ships is: " + str(max))
    print("Max fuel for n ships is: " + str(maxFuel))
    print()

    print("Sum of all fuel is : "+ str(sum(fuels)))
    print("SUm of all times is : " + str(sum(times)))


    return max


def actionToPos(pos, action, Graph):  # convert action to node position
    if (action == outDeg):
        return pos, 0
    #print(Graph[pos].possibleNodes)
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
                    if t % 3 == 0 or j in shipArr:
                        state[j] = ships[j].pos
                        ships[i].otherPos[j] = ships[j].pos
                    state[j] = ships[i].otherPos[j]
                    otherCombTemp = np.arange(len(Graph[ships[i].otherPos[j]].possibleNodes.keys()) - 1).tolist()
                    otherCombTemp.append(outDeg)
                else:
                    otherCombTemp = actionComb[i]
                otherComb.append(otherCombTemp)
            state[i] = ships[i].pos
            # print(otherComb)
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
                            pTemp = actionChoice(i, j, action, 0, qTable, ships, None, pArrOut)[
                                1]  # get p value from decision function
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
                                pTemp = speedChoice(i, j, actionJ[j], speed, qTable, ships, None, pArrOut)[
                                    1]  # get p value from decision function

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

                else:  # if action is stay
                    prox = 0
                    qArrOut[0] = 0
                    qArrOut[j] = 0
                    qArrOut[j + 1] = 0
                    qArrOut[j + 2] = 0
                    qArrOut[j + 3] = 0
                    qArrOut[j + 4] = 0

                if (action == outDeg or qArr[action + 1][0] != 1):

                    qTemp = actionChoice(i, 0, action, 0, qTable, ships, qArrOut, None)[
                        0]  # get action from decision function
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
                    qTemp = speedChoice(i, 0, actionOut[i], speed, qTable, ships, qArrOut, None)[
                        0]  # get speed from decision function
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
    if test < 0:  # QUESTION -- Whats a = 0.6 ?
        a = 0.6
    state = ()
    actionComb = []
    posOut = []

    for ship in ships:
        state = state + (ship.pos,)
        posOutTemp, actionTemp = ship.getPossibleActions(Graph, ships)
        posOut.append(posOutTemp)  # 11 - add potential nodes, actions ships can do
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
        ships[i].movPos = [Graph[ships[i].oldPos].drawLoc[0], Graph[ships[i].oldPos].drawLoc[1]]
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
            # ships[i].seen = []  # Remove sensed node

    # update total time
    for i in range(0, nShip):
        if (ships[i].pos == ships[i].oldPos and ships[i].pos != ships[i].dest):
            ships[i].time += maxTime
    t += 1
    return qTable, ships, t
ts = 0
te = 0


def initPyGame():  # py game run
    global goalSensed
    global seen
    global nShip
    global ts,te
    iterTime = 0

    ts = time.time()
    # init
    Graph = loadGraph()

    # load data
    qTable = np.load("data\qData.npy")
    # nShip = np.shape(qTable)[0]
    ships = initShips(Graph)
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
    show = [0, 0, 0]

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

        xChange = 1 # Use this to do experiments so that you dont have to keep pressing right key
        # if end or collision
        if refresh or t > 400 or collisiontDetect(ships):  # if refresh position or collision
            print("COLLIDED")
            ts = time.time()
            goalSensed = False
            seen = []

            # draw
            #gameDisplay.fill(bgColor)
            #drawGraph(Graph, ships, gameDisplay)
            #drawShips(Graph, ships, gameDisplay, t)
            #pygame.display.update()
            #time.sleep(0)
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

        if endDetPic(ships, Graph):  # if ended at destination
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

        if (
                xChange > 0):
            # take action
            qTable, ships, t = calcNewPos(ships, Graph, qTable, t, 1)

        if iter == 100:
            break


    quit()


if __name__ == "__main__":

    n = 3
    m = 10
    maxDist = 5
    noOfShips = 4
    initPyGame()
