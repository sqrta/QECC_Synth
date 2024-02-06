from adt import *
import copy
import scipy.linalg
import string
from itertools import product
digs = string.digits + string.ascii_letters

class Ket:
    def __init__(self, state, sign = 1) -> None:
        self.state = [int(a) for a in state]
        self.sign = sign
        self.length = len(state)
    
    def MulStabilier(self, stab):
        if isinstance(stab, stabilizer):
            
            if self.length!=stab.length:
                raise ValueError(f"Self length {self.length} does not match stab {stab.length}")
            value = stab.value
            for i in range(self.length):
                if value[i].isTag(("X", "Y")):
                    self.state[i] = abs(self.state[i]-1)
                if value[i].isTag(("Z", "Y")):
                    if self.state[i]==1:
                        self.sign *= -1
        else:
            if self.length!=len(stab):
                raise ValueError("Length does not match")
            # I=0, X=1, Y=2, Z=3
            for i in range(len(stab)):
                if stab[i]==1 or stab[i]==2:
                    self.state[i] = abs(self.state[i]-1)
                if stab[i]==2 or stab[i]==3:
                    if self.state[i]==1:
                        self.sign *= -1            

    def StrValue(self):
        return "".join([str(a) for a in self.state])
    
    def __str__(self) -> str:
        sign = "+" if self.sign == 1 else "-"
        value = self.StrValue()
        return f"{sign}|{value}>"

    def __repr__(self) -> str:
        return str(self)
    
class SuperPositionState:
    def __init__(self, kets) -> None:
        self.kets = kets

    def size(self):
        return len(self.kets[0])

    def length(self):
        return len(self.kets)

    def MulStabilier(self, stab):
        for ket in self.kets:
            ket.MulStabilier(stab)

    def InnerProd(self, other):
        thisMap = self.ketMap()
        result = 0
        otherMap = other.ketMap()
        for key in thisMap.keys():
            if key in otherMap:
                result += thisMap[key] * otherMap[key]
        return result/(self.length() * other.length())**0.5


    def ketMap(self):
        return {a.StrValue():a.sign for a in self.kets}

    def add(self, other):
        self.kets += other.kets

    def sort(self):
        self.kets = sorted(self.kets, key=lambda k: k.StrValue())

    def __str__(self) -> str:
        return "".join([str(a) for a in self.kets])

    def __repr__(self) -> str:
        return str(self)
    
def getLogicalState(stabs):
    n = stabs[0].length
    init = SuperPositionState([Ket('0'*n)])
    for stab in stabs:
        tmp = copy.deepcopy(init)
        tmp.MulStabilier(stab)
        init.add(tmp)
    return init

def ErrorProb(error, px, py, pz):
    errorMap = {0: 1-px-pz-py, 1: px, 2:py, 3:pz}
    return reduce(lambda a,b:a*b, [errorMap[a] for a in error])

def QECMatrix_Depolar(stabs):
    zero = getLogicalState(stabs)
    n = stabs[0].length
    
    paulis = getAllPauliInt(n)
    ex=0.01
    ez=0.05
    px = ex*(1-ez)
    pz = ez*(1-ex)
    py = ex*ez
    N = len(paulis)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            error1 = paulis[i]
            error2 = paulis[j]
            left = copy.deepcopy(zero)
            left.MulStabilier(error1)
            right = copy.deepcopy(zero)
            right.MulStabilier(error2)
            M[i,j] = ErrorProb(error1, px,py,pz)*ErrorProb(error2, px,py,pz)*left.InnerProd(right)
    return M

def NearOptFid(stabs):
    M = QECMatrix_Depolar(stabs)
    np.savetxt("M", M)
    D = copy.deepcopy(np.diagonal(M))
    np.savetxt("D", D)
    N = M.shape[0]
    print(M.shape)
    for i in range(N):
        M[i,i] -= D[i]
    for i in range(N):
        for j in range(N):
            M[i,j] = M[i,j] / (abs(D[i])**0.5+abs(D[j])**0.5)
    return np.linalg.norm(M) 

def allSinglePauli(n):
    result = []
    iden = [0] * n
    result.append(iden)
    def modify(n,i,v):
        iden = [0]*n
        iden[i] = v
        return iden
    for i in range(n):
        for j in range(1,4):
            result.append(modify(n, i, j))
    return result
    
if __name__ == "__main__":
    stabs = ['xxziz', 'zxxzi', 'izxxz', 'zizxx']
    stabs = ['iiixxxx', 'ixxiixx', 'xixixix', 'iiizzzz', 'izziizz', 'ziziziz']
    # stabs = ['xxx', 'zzi']
    stabs = [stabilizer(a) for a in stabs]

    state = getLogicalState(stabs)
    print(state)
    print(state.InnerProd(state))
    import time
    start = time.time()
    F = NearOptFid(stabs)
    print(F)
    end = time.time()
    print(f"use {end-start}s")