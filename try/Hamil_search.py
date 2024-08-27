import numpy as np
from numpy import kron
import functools
from scipy import linalg as LA
from itertools import product
import time
import random
import sys

PAULI_MATRICES = np.array((
    ((0, 1), (1, 0)),
    ((0, -1j), (1j, 0)),
    ((1, 0), (0, -1))
    ))
(X, Y, Z) = PAULI_MATRICES

def tensor(op_list):
    return functools.reduce(kron, op_list, 1)

def pauli2Mat(num_qubits, indexes, paulis):
    '''
    pauli str to numpy matrices
    '''
    op_list = [np.eye(2)] * num_qubits
    for index, pauli in zip(indexes, paulis):
        op_list[index] = pauli
    return tensor(op_list)

def pauliStr2mat(num_qubits, pstrings):
    indexes = []
    paulis = []
    pmap = {'X':X, 'Y':Y, 'Z':Z}
    pauli = pstrings.split('*')
    eff = 1
    if pauli[0].isdigit():
        eff = int(pauli[0])
        pauli.pop(0)
    for p in pauli:
        paulis.append(pmap[p[0].upper()])
        indexes.append(int(p[1:]))
    return eff*pauli2Mat(num_qubits, indexes, paulis)

def pauliExpr2Mat(n, expr):
    """
    n: size
    pstring: e.g. X1*X2 + Z1*Z2
    """
    exp = expr.split('+')
    terms = [PauliTerm(n, e) for e in exp] 
    H = sum([t.value() for t in terms])    
    return H

def vec2Ket(vec):
    args = np.where(np.absolute(vec)>1e-4)[0]
    res = [(vec[a], a) for a in args]

    return res

def ket2Str(n, kets):
    
    strings = [f"({a[0]:.7f})|{a[1]:0{n}b}>" for a in kets]
    return ' + '.join(strings)

class PauliTerm:
    def __init__(self, n, term, eff=1) -> None:
        self.eff = eff
        self.term = term
        self.n = n
    
    def value(self):
        return self.eff*pauliStr2mat(self.n, self.term)
    
    def __str__(self) -> str:
        return f'{self.eff}*{self.term}'

    def __repr__(self):
        return str(self)

    
def ket2Vec(n, kets):
    vec = np.zeros((2**n, 1))
    for ket in kets:
        index = int(ket, base=2)
        vec[index, 0]=1
    return vec

def checkSame(P1, P2, thres = 1e-4):
    nonZ = np.nonzero(np.absolute(P1 - P2)>1e-4)
    if len(nonZ[0])==0:
        return True
    return False

def checkLinear(P1, P2):
    indexes = np.nonzero(np.absolute(P1)>1e-4)
    if len(indexes[0])==0:
        return True
    i1= (indexes[0][0], indexes[1][0])
    if np.absolute(P2[i1]) < 1e-5:
        return False
    r1 = P1[i1] / P2[i1]
    return checkSame(P1, r1*P2)


def testProjector(P, n):
    for i in range(n):
        E = pauliStr2mat(n, f'X{i}')
        res = P @ E @ P
        if not checkLinear(res, P):
            return False
    for i in range(n):
        E = pauliStr2mat(n, f'Z{i}')
        res = P @ E @ P
        if not checkLinear(res, P):
            return False
    for i in range(n):
        E = pauliStr2mat(n, f'Y{i}')
        res = P @ E @ P
        if not checkLinear(res, P):
            return False
    return True

def getP(vecs):
    return sum([vec @ vec.conj().T for vec in vecs])

def bindTerm(n, eff, name):
    X = []
    for i in range(n):
        if eff[i]!=0:
            tmp = f'{name}{i}*{name}{(i+1)%n}'
            if eff[i]==-1:
                tmp = '-'+tmp
            elif eff[i]!=1:
                tmp = f'{eff[i]}*'+tmp
            if eff[i]>0 and i >0:
                tmp = '+'+tmp
            X.append(tmp)
    X = ''.join(X)
    return X

def XZeff2Str(n, Xeff, Zeff):
    print(Xeff)
    print(Zeff)
    X = bindTerm(n, Xeff, 'X')
    Z = bindTerm(n, Zeff, 'Z')

    return X,Z

def phyOpCandiate(n):
    index = [(i,j) for i in range(n-1) for j in range(i, n)]
    res = [f'X{i}' for i in range(n)] + [f'Z{i}' for i in range(n)]
    for ind in index:
        a,b = ind
        res += [f'X{a}*X{b}', f'X{a}*Z{b}',f'Z{a}*X{b}',f'Z{a}*Z{b}']
    return res


def printVecs(n, Xeff, Zeff):
    terms = [PauliTerm(n, f'X{i}*X{(i+1)%n}', Xeff[i]) for i in range(n)] 
    terms += [PauliTerm(n, f'Z{i}*Z{(i+1)%n}', Zeff[i]) for i in range(n)]
    H = sum([t.value() for t in terms])
    eigenvalues, eigenvectors = LA.eigh(H)

    index = np.absolute(eigenvalues)<1e-6
    eigenvectors = eigenvectors[:, index]
    
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        print(vec.shape)
        print('-----------')
        print('-----------')
        # print(H @ vec.T)
        print(ket2Str(n, vec2Ket(vec)))

def getHamil(n, Xeff, Zeff):
    terms = [PauliTerm(n, f'X{i}*X{(i+1)%n}', Xeff[i]) for i in range(n)] 
    terms += [PauliTerm(n, f'Z{i}*Z{(i+1)%n}', Zeff[i]) for i in range(n)]
    # print(terms)
    H = sum([t.value() for t in terms])
    return H

def getSpace(n, H):
    eigenvalues, eigenvectors = LA.eigh(H)
    index = np.absolute(eigenvalues)<1e-6
    eigenvectors = eigenvectors[:, index]
    return eigenvectors

def testH(n, H):
    eigenvectors = getSpace(n, H)
    PList = []
    # print(f"space size: {eigenvectors.shape[1]}")
    if eigenvectors.shape[1]<4:
        return False, eigenvectors.shape[1]
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        vec = vec.reshape(len(vec), 1)
        PList.append(vec)
    # PList = [ket2Vec(n, ['1000', '0111']), ket2Vec(n, ['0100', '1011']), ket2Vec(n, ['0010', '1101']), ket2Vec(n, ['0001', '1110'])]
    P = getP(PList)
    result = testProjector(P, n)
    # if result:
    #     print(eigenvectors.shape[1])
    return result, eigenvectors.shape[1]

def getProjector(n, H):
    eigenvectors = getSpace(n, H)
    PList = []
    # print(f"space size: {eigenvectors.shape[1]}")
    if eigenvectors.shape[1]<4:
        return False, eigenvectors.shape[1]
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        vec = vec.reshape(len(vec), 1)
        PList.append(vec)
    # PList = [ket2Vec(n, ['1000', '0111']), ket2Vec(n, ['0100', '1011']), ket2Vec(n, ['0010', '1101']), ket2Vec(n, ['0001', '1110'])]
    P = getP(PList)
    return P

def testLogicalOp(n, pauliStr, H, Pc):
    op = pauliExpr2Mat(n, pauliStr)
    P = getProjector(n, H)
    O = P @ op @ P
    return commuteOrNot(O, Pc)

def commuteOrNot(P1, P2, sign=1):
    """
    test P1@P2 - sign * P2@P1
    """
    M = P1 @ P2 - sign * P2 @ P1
    if LA.norm(M) < 1e-4:
        return True
    return False

def testEff(n, Xeff, Zeff):
    terms = [PauliTerm(n, f'X{i}*X{(i+1)%n}', Xeff[i]) for i in range(n)] 
    terms += [PauliTerm(n, f'Z{i}*Z{(i+1)%n}', Zeff[i]) for i in range(n)]
    # print(terms)
    
    H = sum([t.value() for t in terms])
    return testH(n, H)


def searchHpen(n, k, thres=0, path = 'result'):
    with open(path, 'w') as f:
    # if True:
        Xeff_can = list(product(range(-k, k+1), repeat=n))
        Zeff_can = list(product(range(-k, k+1), repeat=n))
        random.shuffle(Xeff_can)
        random.shuffle(Zeff_can)
        for Xeff in Xeff_can:
            if Xeff[0]<0:
                continue
            print(Xeff)
            for Zeff in Zeff_can:
                if Zeff[0]<0:
                    continue
                res, size = testEff(n, Xeff, Zeff)
                if res and size>=thres:
                    f.write((f"succeed: {Xeff}, {Zeff}, size: {size}, {XZeff2Str(n, Xeff, Zeff)}\n"))
                    # print((f"succeed: {Xeff}, {Zeff}, size: {size}"))




if __name__ =='__main__':
    n = 6

    
    depth = 3
    thres = 0
    if len(sys.argv)>1:
        n=int(sys.argv[1])
    if len(sys.argv)>2:
        depth = int(sys.argv[2])
    if len(sys.argv)>3:
        thres = int(sys.argv[3])
    # c1 = ket2Vec(n, ['1000', '0111']) 
    # P = c1 @ c1.conj().T
    # print(P)
    # PList = [ket2Vec(n, ['1000', '0111']), ket2Vec(n, ['0100', '1011']), ket2Vec(n, ['0010', '1101']), ket2Vec(n, ['0001', '1110'])]
    # P = getP(PList)
    # E = pauliStr2mat(n, f'X{0}*X1+Z0*Z2')
    # res = P @ E @ P
    # print(checkLinear(res, P))
    # exit(0)
    start = time.time()


    # Xeff = sum([[0,1] for i in range(n//2)], start = [])
    # Zeff = sum([[0,-1] for i in range(n//2)], start=[])
    # print(Xeff)
    # print(Zeff)
    # printVecs(n, Xeff, Zeff)
    # res = testEff(n, Xeff, Zeff)
    # print(res)
    # exit(0)
    # res = testEff(n, Xeff, Zeff)
    # print(res)
    searchHpen(n, depth, thres, f'result{n}_{depth}_{thres}')
    end = time.time()
    print(f'use {end-start}s')