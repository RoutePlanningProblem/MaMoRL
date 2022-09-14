# Baseline-2


# 1) Layout of folder:

In this folder, there are following subfolders:


a) Modules -- file that contains 3 reward functions (They are being used in Train.py).

b) Train.py -- file to train the model (It needs to be run before Run.py).

c) Run.py -- file to run the model to get the desired output.

d) decision.py -- file that is used inside other functions for acion and speed selection.

e) inputs -- folder that contains all the input grids.


# 2) Train.py Inputs

You can change the following parameter values:

1) Number of ships --> line 10

2) MaxSpeed of ship --> line 11

3) Number of Epoch --> line 12

4) Max OutDeg of node --> line 16

5) Graph file name --> line 17 (Requires a .csv file at "./inputs" directory)

6) Number of iterations --> line 140


# 3) Train.py Output

Output of training will be set of Q tables stored in "./data" subdirectory.

# 4) Run.py Inputs

You can change the following parameter values and please make sure that these parameters value be consitant to their corresponding value in Train.py:

1) Number of ships --> line 15

2) Max OutDeg of node --> line 16

3) Graph file name --> line 17 (Requires a .csv file at "./input" directory)

4) Result file name --> line 18 (Name the .csv file to store the results)

5) Communication Frequency --> line 19

6) MaxSpeed of ship --> line 112

7) Start Location of ships for 10 runs --> line 22
(To have a fair comparison, we used exactly the same starting locations from file result.csv in Approx-MaMoRL/Approx-MaMoRL_PartialKnowledge)

8) Destination of ships for 10 runs --> line 34
(To have a fair comparison, we used exactly the same destination from file result.csv in Approx-MaMoRL/Approx-MaMoRL_PartialKnowledge)


# 5) Run.py Output

This code will be run for 10 iterations and store the results in the .csv file  that is defined in line 19.  Information on the number of ships, their starting points, destinations, and number of neighbors, as well as objective values like the maximum time and total fuel consumption, are among the output that are stored. It also saves running time for the code and the size of q tables. Some of these information will be print out and if a collision occurs, it will note that as well.



# 6) Remark

** First need to run Train.py to train the model and obtain the data that is needed to run Run.py **

** Average of these 10 runs will be considered as the final value for the specified combination **

** if a few actions has the same reward, one of them will be chosen randomly, so because of the randomness that is involved, the results may be slightly different from what we presented in our paper **

** If a collision occurs for a certain case, "NaN" will be saved in the result.csv file **

** Due to its nature, this method is prone to collisions most of the times **
