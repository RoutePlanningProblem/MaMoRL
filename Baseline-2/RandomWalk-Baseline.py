


import numpy as np
import pandas as pd
import networkx as nx
import csv
import random
import time
import math


outDeg=10
nShips=6
noNodes=704
maxSpeed = [5 for _ in range(nShips) ]
sensingRadius=[1, 1, 1, 1, 1, 1]
goalNode= 464
crashed=False



def loadGraph():  # load graph from csv

    global outDeg
    with open("input/704nodes_1455edges_degree10.csv") as csvfile:
        rawData = list(csv.reader(csvfile, delimiter=','))
    rawData.pop(0)

    retLink = -np.ones((len(rawData), outDeg),dtype=np.int64)  # Links from a node
    retDist = -np.ones((len(rawData), outDeg),dtype=np.int64)  # Weights of Links
    retMax = np.full(len(rawData), outDeg-1,dtype=np.int64)  # Outdegree of a node
    rawPoints = -np.ones(len(rawData),dtype=np.int64)  # Storing the nodes in raw node ID

    n = 0

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


    return retLink, retDist, retMax



graphLink, graphDist, graphMax= loadGraph()


#convert data to a grid

G = nx.DiGraph()
G.add_nodes_from([i for i in range(noNodes)])

edges=[]
for i in G.nodes():
    for j in graphLink[i]:
        if (j!=-1):
            edges.append((i,j))
            

G.add_edges_from(edges)

weights={}
for edge in G.edges():
    weights[edge]=graphDist[edge[0]][np.where(graphLink[edge[0]]==edge[1])[0][0]]
    
for e in G.edges():
    G[e[0]][e[1]]['weight'] = weights[e]



# initializing the start position for assets

startPoints=random.sample(G.nodes(), nShips)

print("Ships Starting Point=", startPoints)
print("Destination=", goalNode)


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

#print out objective values

def printMaxTime(times, fuels):  # print time at end
    global maxTime
    maxTime = 0
    
    for i in range(nShips):
        if (times[i] > maxTime):
            maxTime = times[i]
 
    maxTime = round(maxTime, 4)

    print()
    print("Max time for n ships is: " + str(maxTime))
    print()

    print("Sum of all fuel is : "+ str(sum(fuels)))


start_time = time.time()
ships=[]
order=[]
current=startPoints
times=[0]*nShips
fuels=[0]*nShips
wayPoints=[[current[i]] for i in range(nShips)]

sensed={i:[] for i in range(nShips)}
while not crashed:


    nextPos, speed=nextMove(G, current)

    sensed=sensedNodes(G, current, sensed)


    for i in range(nShips):
        for j in range(nShips):
            if i!=j:
                if nextPos[i]==nextPos[j]:
                    print("collision")
                    crashed=True

    for i in range(nShips):

        wayPoints[i].append(nextPos[i])

        if goalNode in sensed[i]:
            ships.append(i)
            for key in sensed:
                if key!=i:
                    sensed[key].append(goalNode)
            wayPoints[i]=wayPoints[i]+nx.dijkstra_path(G, source=nextPos[i], target=goalNode , weight='weight')
            distance=nx.dijkstra_path_length(G, source=nextPos[i], target=goalNode , weight='weight')
            times[i]+=  distance/speed[i]
            fuels[i]+= times[i] * (0.2525 * math.pow((speed[i] + 10) * 2, 2) - 1.6307 * (speed[i] + 10) * 2)
            crashed=True
            print("ship", str(i) , "Reached Goal")
            order.append(str(i))



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
            print("ship", str(i) , "Reached Goal")                
            order.append(str(i))

printMaxTime(times, fuels)

print()
print("--- %s seconds ---" % (time.time() - start_time))


#Store results in a csv file

with open('random_walk/degree/10.csv', mode='a') as csv_file:
                    fieldnames = ['No of Ships', 'OUT DEGREE', 'NODE INDEX', 'Ship Location', 'Max time', 'Total Fuel', 'Total time taken', 'Running time', 'Order of Ships']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    if order!=[]:
                        if csv_file.tell() == 0:
                            writer.writeheader()
                        writer.writerow({'No of Ships': nShips, 'OUT DEGREE': outDeg, 'NODE INDEX': goalNode, 'Ship Location':startPoints, 'Max time':str(maxTime), 'Total Fuel':str(sum(fuels)), 'Total time taken':str(sum(times)), 'Running time':str(time.time() - start_time), 'Order of Ships':order})




