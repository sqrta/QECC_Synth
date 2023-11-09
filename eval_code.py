from adt import *
import itertools
from enumerator import *

stab_group = stabilizer_group(code513)
print(len(stab_group),stab_group)
a=stabilizer('ixzix')
b=stabilizer("xzzxi")
def allError(error, code):
    stab_group = stabilizer_group(code)
    return [(a*error).toInt() for a in stab_group]

def getPi(S):
    length = len(S)
    return sum([a.toMatrix() for a in S])/length
def PiEs(S, Es):
    E_matrix = Es.toMatrix()
    length = len(S)
    

e=stabilizer('ixzix')
result = allError(e, code513)
print(len(getAllPauli(2)))

