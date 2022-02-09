import numpy as np

def speedChoice(i, j, action, speed, qTable, pTable, qArr, pArr): # change this to change decision for action
    qReturn = 0.0
    pReturn = 0.0
    try:
        qReturn = np.sum(qTable[i][1][1] * qArr)
    except:
        None

    try:
        pReturn = np.sum(pTable[i][j][1][1] * pArr)
    except:
        None

    return qReturn, pReturn

def actionChoice(i, j, action, speed, qTable, ships, qArr, pArr): # change this to change decision for action
    qReturn = 0.0
    pReturn = 0.0
    try:
        qReturn = np.sum(qTable[i][0][0] * qArr)
    except:
        None

    try:
        pReturn = np.sum(ships[i].pTable[j][0][0] * pArr)
    except:
        None

    return qReturn, pReturn
