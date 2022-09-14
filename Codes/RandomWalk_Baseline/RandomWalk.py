
import numpy as np
import pandas as pd
import networkx as nx
import csv
import random
import time as time
import math
import sys
from statistics import *
from scipy.spatial import distance

nShips = 2  # Set number of ships here
outDeg = 7 #Set the MAX outdeg here
noOfIter = 10 #Number of iterations for each set of data.
noNodes = 704 # Number of nodes in the grid
maxSpeed = np.array ([5]*nShips) #Set the max speed here, np array size must equal number of ships

sensingRadius = [1]*nShips # ships' sensing radius
# communicationInterval = 1 # ships' communication Interval
filename = 'inputs/Varying_Degree/704nodes_1399edges_degree7.csv' #Set input grid file here
resultPath='results.csv' #Name of file with input grid

#Set of starting of points to be defined here
starts = [[237, 111],
[315, 104],
[344, 371],
[586, 197],
[22, 174],
[644, 440],
[656, 96],
[592, 131],
[562, 231],
[582, 367]]

#Corrosponding goal nodes to be defined here
goalNodes = [56,
435,
75,
279,
162,
158,
155,
224,
680,
487]


def loadGraph():  # load graph from csv

    global outDeg
    with open(filename) as csvfile:
        rawData = list(csv.reader(csvfile, delimiter=','))
    rawData.pop(0)

    retLink = -np.ones((len(rawData), outDeg), dtype=np.int64)  # Links from a node
    retDist = -np.ones((len(rawData), outDeg), dtype=np.int64)  # Weights of Links
    retMax = np.full(len(rawData), outDeg - 1, dtype=np.int64)  # Outdegree of a node
    rawPoints = -np.ones(len(rawData), dtype=np.int64)  # Storing the nodes in raw node ID
    lat_long = {i: [] for i in range(noNodes)} #store nodes lat/long

    n = 0
    k = 0

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

    return retLink, retDist, retMax, lat_long


graphLink, graphDist, graphMax, lat_long = loadGraph()

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

G, weights = buildGraph(noNodes, graphLink, graphDist, lat_long)

# Update sensed nodes by ships
def sensedNodes(G, current, sensed):

    for i in range(nShips):

        sensed[i]=list(set(sensed[i]+list(G.neighbors(current[i]))))
        if sensingRadius[i]!=1:
            for j in sensed[i]:
                sensed[i]=list(set(sensed[i]+list(G.neighbors(j))))          
    return sensed


#Choose next action and speed

def nextMove(G,current):
    
    nextPos=[]
    speed=[]
    for i in range(nShips):
        nextPos.append(random.choice(list(G.neighbors(current[i]))))
        speed.append(random.randint(1, maxSpeed[i]))
    return nextPos, speed



def printMaxTime(times, fuels):  # print time at end
    
    global maxTime

    maxTime = 0
    
    for i in range(nShips):
        if (times[i] > maxTime):
            maxTime = times[i]
 
    maxTime = round(maxTime, 4)

    print()
    print("Max time for n ships is: " + str(maxTime))
    print("Sum of all fuel is : "+ str(sum(fuels)))
    
    return maxTime , sum(fuels)



def main(startPoints): # main function

    global nShips, maxTime, totalFuel

    # start times
    start_time = time.time()
    ships=[]
    order=[]
    crashed=False
    current=startPoints
    times=[0]*nShips
    fuels=[0]*nShips
    wayPoints=[[current[i]] for i in range(nShips)] # store the waypoints for each ship's route

    sensed={i:[] for i in range(nShips)}
    
    while not crashed: # continue until ships are not collided

        # select ships' action and speed randomly
        nextPos, speed=nextMove(G, current)

        # update sensed nodes
        sensed=sensedNodes(G, current, sensed)

        # check if any two ships have the same locations in the next step
        for i in range(nShips):
            for j in range(nShips):
                if i!=j:
                    if nextPos[i]==nextPos[j]:
                        print("collision")
                        crashed = True
                        return True
        if crashed:
            return True
            exit()

        for i in range(nShips):

            wayPoints[i].append(nextPos[i])

            # if destination is sensed, Dijkstra will be used to reach destination from current position
            if goalNode in sensed[i]:
                ships.append(i)
                for key in sensed:
                    if key!=i:
                        sensed[key].append(goalNode)
                wayPoints[i]=wayPoints[i]+nx.dijkstra_path(G, source=nextPos[i], target=goalNode , weight='weight')
                distance=nx.dijkstra_path_length(G, source=nextPos[i], target=goalNode , weight='weight')
                times[i]+=  distance/speed[i]
                fuels[i]+= times[i] * (0.2525 * math.pow((speed[i] + 10) * 2, 2) - 1.6307 * (speed[i] + 10) * 2)
                print("ship", str(i) , "Reached Goal")
                order.append(str(i))
                crashed = True

            # Update time and fuel
            times[i]+=  weights[(current[i], nextPos[i])]/speed[i]
            fuels[i]+= times[i] * (0.2525 * math.pow((speed[i] + 10) * 2, 2) - 1.6307 * (speed[i] + 10) * 2)

        current=nextPos

    for i in range(nShips):
        if i not in ships:
            if goalNode in sensed[i]:
                wayPoints[i]=wayPoints[i]+nx.dijkstra_path(G, source=nextPos[i], target=goalNode , weight='weight')
                distance=nx.dijkstra_path_length(G, source=nextPos[i], target=goalNode , weight='weight')
                times[i]+=  distance/speed[i]
                fuels[i]+= times[i] * (0.2525 * math.pow((speed[i] + 10) * 2, 2) - 1.6307 * (speed[i] + 10) * 2)
                order.append(str(i))


    # print objective values
    maxTime, totalFuel = printMaxTime(times, fuels)
    # end time
    end_time = time.time()

    print("--- %s seconds ---" % (end_time - start_time))


# Code for automated running
for m in range(noOfIter):

    goalNode = goalNodes[m]  # Set the node index you want as the goal
    start = starts[m] #starting locations

    failed = 1
    failed = main(start)

    if failed:
        with open(resultPath, mode='a') as csv_file:
            fieldnames = ['No of Ships', 'out degree', 'Goal Node',  'Ship Location', 'Max time', 'Total Fuel']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow(
                {'No of Ships': nShips, 'out degree': outDeg, 'Goal Node': goalNode, 'Ship Location': str(start), 'Max time': 'collided', 'Total Fuel': 'Collided'})

    else:
        with open(resultPath, mode='a') as csv_file:
            fieldnames = ['No of Ships', 'out degree', 'Goal Node', 'Ship Location',  'Max time', 'Total Fuel']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow({'No of Ships': nShips, 'out degree': outDeg, 'Goal Node': goalNode, 'Ship Location': str(start), 'Max time': str(maxTime ), 'Total Fuel': str(totalFuel)})