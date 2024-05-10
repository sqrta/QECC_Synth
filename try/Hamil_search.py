import numpy as np
from numpy import kron
import functools
from scipy import linalg as LA
import time
import random

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
    for p in pauli:
        paulis.append(pmap[p[0].upper()])
        indexes.append(int(p[1:]))
    return pauli2Mat(num_qubits, indexes, paulis)

def vec2Ket(vec):
    args = np.where(np.absolute(vec)>1e-4)[0]
    res = [(vec[a], a) for a in args]

    return res

def ket2Str(n, kets):
    
    strings = [f"({a[0]:.3f})|{a[1]:0{n}b}>" for a in kets]
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

if __name__ =='__main__':
    n = 4
    Xeff = [1,1.5,1,1.5]
    Zeff = [1,0.5,1,0.5]
    terms = [PauliTerm(n, f'X{i}*X{(i+1)%n}', Xeff[i]) for i in range(n)] 
    terms += [PauliTerm(n, f'Z{i}*Z{(i+1)%n}', Zeff[i]) for i in range(n)]
    print(terms)
    start = time.time()
    H = sum([t.value() for t in terms])
    eigenvalues, eigenvectors = LA.eigh(H)
    # print(H.shape)

    # tmp_vec = ket2Vec(n, ['0010', '1101'])
    # print(H @ tmp_vec) 
    # exit(0)
    # print(eigenvectors)
    # index = np.argsort(np.absolute(eigenvalues))
    
    # print(eigenvalues)
    index = np.absolute(eigenvalues)<1e-6
    eigenvectors = eigenvectors[:, index].T
    # print(index)
    # print(eigenvalues[index])
    # print(eigenvectors)    
    # exit(0)
    # print(eigenvectors[0] @ eigenvectors[2].T)
    for vec in eigenvectors:
        print('-----------')
        print('-----------')
        # print(H @ vec.T)
        print(ket2Str(n, vec2Ket(vec)))
    end = time.time()
    print(f'use {end-start}s')