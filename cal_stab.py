import numpy as np
from numpy import kron
import functools

def tensor(op_list):
    return functools.reduce(kron, op_list, 1)

X = np.array([[0,1],
              [1,0]])

Z = np.array([[1,0],
              [0,-1]])

Y = 1j * X @ Z

I = np.array([[1,0],
              [0,1]])

n = 2
init = np.array([1,0,0,1j])
paulis = {'X':X, 'Y': Y, 'Z':Z, 'I': I}
print(tensor([X,I]))
for i in paulis.keys():
    for j in paulis.keys():
        print(i,j)
        m = tensor([paulis[i],paulis[j]])
        tmp = m.dot(init)
        print(init.conj().T @ tmp)
