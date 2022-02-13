import numpy as np


qTable = np.load("qData.npy")

# #https://www.geeksforgeeks.org/find-the-memory-size-of-a-numpy-array/
print("Q Array size is "+str(qTable.itemsize * qTable.size) +" bytes" )

print(qTable.size)

totalPTableSize = 0

for i in range(0, 2):

    for j in range(0, 2):
        if (i != j):
            pArray = np.load(""+str(i) + '-' + str(j) + '-' + 'pData.npy')
            print(pArray.size)
            totalPTableSize += pArray.itemsize * pArray.size



print("Total P table size is " + str(totalPTableSize) + " bytes")
