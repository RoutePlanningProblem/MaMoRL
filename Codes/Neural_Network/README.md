# NN-Approx-MaMoRL


# 1) Layout of folder:

In this folder, there are following subfolders:


a) Modules -- file that contains 3 reward functions (They are being used in Train.py).

b) TrainNN.py -- file to train the model (It needs to be run before Run.py).

c) Run.py -- file to run the model to get the desired output.

d) decision.py -- file that is used inside other functions for acion and speed selection.

e) inputs -- folder that contains all the input grids.

f) NN_Models -- folder that contains the trained model for p and q tables.

g) data -- folder that has required data for Neural Network training.


# 2) TrainNN.py Inputs

You can change the following parameter values:

1) Train data directory --> line 19

2) Test size --> line 58

3) Number of Epochs and batch size --> line 61

4) Name the trained model for saving --> line 89


# 3) TrainNN.py Output

Output of training will be a trained model for predicting P/Q values stored in "./NN_Models" subdirectory.


# 4) Run.py Inputs

You can change the following parameter.

1) Number of ships --> line 15

2) Max OutDeg of node --> line 16

3) Destination index --> line 17

4) Graph file name --> line 22 (Requires a .csv file at "./input" directory)

5) Directory to the trained models --> line 359, 360 & 361

6) Communication Frequency --> line 24

7) MaxSpeed of ship --> line 96



# 5) Run.py Output

This code will print out information on starting points and objective values like the maximum time and total fuel consumption as well as running time for the code.


# 6) Remark

** First need to run TrainNN.py to obtain the trained the model that is needed to run Run.py **

** if a few actions has the same reward, one of them will be chosen randomly, so because of the randomness that is involved, the results may be slightly different from what we presented in our paper **

** In each run, the starting locations are chosen randomly. As a result in certain cases, a few runs are needed for a successful run and avoid collisions **

