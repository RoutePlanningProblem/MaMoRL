# MaMoRL



# 1) Setup

You can find the required dependencies to run all the codes in the file "Required_Dependencies.txt".

To run code, import entire folder into IDE. PyCharm is one option. Import as a project 

# 2) Input Graph

Requires a .csv file at `input` directory that is in the following format:

```
NodeID, Latitude, Longitude, Neighbor1, Distance1, Neighbor2, Distance2, Neighbor3, Distance3, Neighbor4, Distance4, Neighbor5, Distance5, .... Neighbor N, Distance N
```


# 3) Layout of folders

There are three folders:

1. Plots : Contains all the plots even the ones that are not presented in the draft.
2. Grids : Contains all the synthetic and real-world grids.
3. Codes : Contains different folders for each algorithm:

    a) Exact Model (MaMoRL)

    b) Approx Model (Approx-MaMoRL)

    c) Approx-MaMoRL with Partial Knowledge

    d) Neural Network Model  (NN-Approx-MaMoRL)

    e) Baseline-1

    f) Baseline-2

    g) Random Walk Baseline

In each folder, there is subfolders:


a) Modules -- file with reward functions 

b) Train.py -- file to train the model 

c) run.py -- file to run the model. Model will store the results in a csv file


# 4) Changing parameter values 

You can change parameter values through the respective run and train file at the top. 

You can change the following :

1) MaxSpeed of ship 

2) Max OutDeg of node

3) Input file name 

4) Result file name

5) Epoch value (in Train.py file)

6) iterations value (in Train.py file)

7) Number of ships

# 5) Output

Output of training will be set of P/Q tables stored in Data subdirectory

Output of run will be information on the ship actions, including time and fuel taken. If a collision occurs, it will note that as well. 

# Remark
Please note exact model, due to system constraints, will not run for more than the parameters as below:
```
2 ships, 400 nodes, outDeg 6
```
** Certain cases, a few runs are needed for a successful run **
