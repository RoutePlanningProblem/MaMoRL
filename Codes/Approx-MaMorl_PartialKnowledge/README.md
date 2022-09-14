# Approx-MaMoRL_PartialKnowledge


# 1) Layout of folder:

In this folder, there are following subfolders:


a) Modules -- file that contains 3 reward functions (They are being used in Train.py).

b) Train.py -- file to train the model (It needs to be run before Run.py).

c) Run.py -- file to run the model to get the desired output.

d) decision.py -- file that is used inside other functions for acion and speed selection.

e) inputs -- folder that contains all the input grids.


# 2) Train.py Inputs

You can change the following parameter values:

1) Number of ships --> line 12

2) MaxSpeed of ship --> line 13

3) Number of Epoch --> line 14

4) Max OutDeg of node --> line 18

5) Graph file name --> line 19 (Requires a .csv file at "./inputs" directory)

6) Number of nodes --> line 20

7) Number of nodes in the known region --> line 21 (It should be 60% of number of nodes)

8) Number of iterations --> line 150


# 3) Train.py Output

Output of training will be set of P/Q tables stored in "./data" subdirectory.

# 4) Run.py Inputs

You can change the following parameter values and please make sure that these parameters value be consitant to their corresponding value in Train.py:

1) Number of ships --> line 21

2) Max OutDeg of node --> line 22

3) Graph file name --> line 25 (Requires a .csv file at "./input" directory)

4) Result file name --> line 26 (Name the .csv file to store the results)

5) Number of nodes in graph --> line 27

6) Number of nodes in the known region --> line 28 (It should be 60% of number of nodes)

7) Communication Frequency --> line 29

8) MaxSpeed of ship --> line 24 & line 298



# 5) Run.py Output

This code will be run for 10 iterations and store the results in the .csv file  that is defined in line 26.  Information on the number of ships, their starting points, destinations, and number of neighbors, as well as objective values like the maximum time and total fuel consumption, are among the output that are stored. It also saves running time for the code and the size of p and q tables. Some of these information will be print out and if a collision occurs, it will note that as well. 



# 6) Remark

** First need to run Train.py to train the model and obtain the data that is needed to run Run.py **

** Average of these 10 runs will be considered as the final value for the specified combination **

** if a few actions has the same reward, one of them will be chosen randomly, so because of the randomness that is involved, the results may be slightly different from what we presented in our paper **

** In each run, the destination and the starting locations are chosen randomly. As a result in certain cases, a few runs are needed for a successful run and avoid collisions **

