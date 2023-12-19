from adt import *
import itertools
from enumerator import *
import time
from progEval import *


# stab_group = stabilizer_group(code513)
# print(len(stab_group),stab_group)
# a=stabilizer('ixzix')
# b=stabilizer("xzzxi")
def allError(error, code):
    stab_group = stabilizer_group(code)
    return [(a*error).toInt() for a in stab_group]

def getPi(S):
    length = float(len(S))
    return sum([a.toMatrix() for a in S])/length
def PiEs(S, Es):
    E_matrix = Es.toMatrix()
    length = len(S)
    return sum([a.toMatrix() for a in S])/length

def stabilizers2Int(stabilizers):
    res = []
    for stabs in stabilizers:
        res.append(stabs.toInt())
    return res

def B(n, M1, M2):
    sum = 0
    for e in getAllPauliInt(n):
        E = intVec2M(e, n)
        M = E @ M1 @ E.conj() @ M2
        res = np.trace(M)
        if res!=0:
            sum+=1
    return sum
def A(n, M1, M2):
    sum = 0
    for e in getAllPauliInt(n):
        E = intVec2M(e, n)
        res = np.trace(E @ M1)*np.trace(E.conj() @ M2)
        if res!=0:
            sum+=1

    return sum

def MaxLikeDecodeEval(stabilizers, k):
    n = next(iter(stabilizers)).length


if __name__ == "__main__":

    px = 0.001
    pz = 0.005
    stabs = stabilizers2Int(code513.stabs)
    print(stabs)
    candidateProg = [prog513_44, prog13_1_4_99, prog913_4_4, prog913_6_6, prog913_8_7, prog713_4_4, prog713_5_5, prog713_6_6, prog623_trick, prog6_1_3_54, prog823_43]
    candidateProg = [prog513_44, prog6_1_3_54, prog7_3_66, prog813_66, prog913_6_6, prog1013_6_7, prog11_1_3_98,prog12_3_108, prog13_1_4_1010]
    candidateProg = [prog12_3_106]
    varList = []
    for prog in candidateProg:
        cm = prog2CM(prog)
        stabs = [stab.toInt() for stab in checkM2Stabilizers(cm.matrix)]
        n = cm.matrix.shape[1]//2
        xe, ze = ErrorFrCode(px, pz, cm)
        print(f"code_{n}=CodeWithError({stabs},{xe},{ze})")
        varList.append(f"code_{n}")
    print(f"candidate=[{','.join(varList)}]")
