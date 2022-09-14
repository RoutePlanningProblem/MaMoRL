# Exact-MaMoRL


# 1) Layout of folder:

In this folder, there are following subfolders:


a) Modules -- file that contains 3 reward functions (They are being used in Train.py).

b) Train.py -- file to train the model (It needs to be run before Run.py).

c) Run.py -- file to run the model to get the desired output.

d) sizescript.py -- file that is used for calculating the size of P and Q tables.

e) inputs -- folder that contains all the input grids.


# 2) Train.py Inputs

You can change the following parameter values:

1) Number of ships --> line 26

2) Number of nodes in the grid --> line 27

3) Max OutDeg of node --> line 28

4) MaxSpeed of ship --> line 35

5) Graph file name --> line 37 (Requires a .csv file at "./inputs" directory)

6) Number of epochs --> line 273


# 3) Train.py Output

Output of training will be set of P/Q tables stored in "./data" subdirectory.


# 4) Run.py Inputs

You can change the following parameter values and please make sure that these parameters value be consitant to their corresponding value in Train.py:

1) Number of ships --> line 17

2) Max OutDeg of node --> line 18

3) Number of nodes in graph --> line 25

4) Graph file name --> line 23 (Requires a .csv file at "./inputs" directory)

5) Result file name --> line 24 (Name the .csv file to store the results)

6) Communication Frequency --> line 26

7) MaxSpeed of ship --> line 84



# 5) Run.py Output

This code will be run for 10 iterations and store the results in the .csv file  that is defined in line 24.  Information on the number of ships, their starting points, destinations, and number of neighbors, as well as objective values like the maximum time and total fuel consumption, are among the output that are stored. It also saves running time for the code and the size of p and q tables. Some of these information will be print out and if a collision occurs, it will note that as well. 



# 6) Remark

Please note that the Exact-MaMoRL, due to system constraints, will not run for more than the parameters as below:

```
2 ships, 400 nodes, outDeg 6

```

** First need to run Train.py to train the model and obtain the data that is needed to run Run.py **

** Average of these 10 runs will be considered as the final value for the specified combination **

** if a few actions has the same reward, one of them will be chosen randomly, so because of the randomness that is involved, the results may be slightly different from what we presented in our paper **

** In each run, the destination and the starting locations are chosen randomly. As a result in certain cases, a few runs are needed for a successful run and avoid collisions **

