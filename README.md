# MaMoRL



# 1) Setup

You will need the following dependencies:

- Python 3
- Pygame 2.0.0.dev6
- numba 0.50.1
- matplotlib 3.3.1
- numpy 1.18.5

To run code, import entire folder into IDE. PyCharm is one option. Import as a project 

# 2) Input Graph

Requires a .csv file at `input` directory that is in the following format:

```
NodeID, Latitude, Longitude, Neighbor1, Distance1, Neighbor2, Distance2, Neighbor3, Distance3, Neighbor4, Distance4, Neighbor5, Distance5, .... Neighbor N, Distance N
```


# 3) Layout of folders

There are three folders for three different models:
a) Exact Model (MaMoRL)

b) Approx Model (Approx-MaMoRL)

c) Baseline 

In each folder, there is subfolders:

a) data -- Where P/Q tables will be stored 

b) Modules -- files with reward functions 

c) input -- place your input file here (.csv) 

d) Train.py -- file to conduct training 

e) run.py -- file to run the model. Model will print out needed information


# 4) Changing parameter values 

You can change parameter values through the respective run and train file at the top. 

You can change the following :

1) MaxSpeed of ship 

2) Max OutDeg of node

3) Input file name 

4) Epoch value (in Train.py file)

5) No of ships 

6) SmallerGraph -- True/False (See code comments for details) 

# 5) Output

Output of training will be set of P/Q tables stored in Data subdirectory

Output of run will be information on the ship actions, including time and fuel taken. If a collision occurs, it will note that as well. 

# Remark
Please note exact model, due to system constraints, will not run for more than the parameters as below:
```
2 ships, less than 700 nodes, outDeg 6
```
** Certain cases, a few runs are needed for a successful run **
