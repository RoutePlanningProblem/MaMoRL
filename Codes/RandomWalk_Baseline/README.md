# RandomWalk_Baseline


# 1) Layout of folder:

In this folder, there are following subfolders:


a) RandomWalk.py

b) inputs -- folder that contains all the input grids.


# 2) Inputs

You can change the following parameter values:

1) Number of ships --> line 13

2) Max OutDeg of node --> line 14

3) Number of nodes in the grid --> line 16

4) MaxSpeed of ship --> line 17

5) Graph file name --> line 22 (Requires a .csv file at "./input" directory)

6) Result file name --> line 23 (Name the .csv file to store the results)

7) Start Location of ships for 10 runs --> line 26
(To have a fair comparison, we used exactly the same starting locations from file result.csv in Approx-MaMoRL/Approx-MaMoRL_PartialKnowledge)

8) Destination of ships for 10 runs --> line 38
(To have a fair comparison, we used exactly the same destination from file result.csv in Approx-MaMoRL/Approx-MaMoRL_PartialKnowledge)


# 3) Outputs

This code will be run for 10 iterations and store the results in the .csv file  that is defined in line 23.  Information on the number of ships, their starting points, destinations, and number of neighbors, as well as objective values like the maximum time and total fuel consumption, are among the output that are stored. Some of these information will be print out and if a collision occurs, it will note that as well.


# 4) Remark

** Average of these 10 runs will be considered as the final value for the specified combination **

** If a collision occurs for a certain case, "NaN" will be saved in the result.csv file **