import numpy as np
T,S,C,M,UP = 0,0,0,0,1000
CPU = np.array([])
ZERO = np.array([])
RETURN_PMS = False
RECORD_ENLARGE = False

def InitialGlobalVars(T_temp:int, UP_temp:int=1000):
    global T,S,C,M,UP,CPU,ZERO
    T,S = T_temp,3
    C = 2**(T+1)
    M = 2*C
    CPU = np.array([2**i for i in range(T)])
    ZERO = np.array([0]*T)
    UP = UP_temp