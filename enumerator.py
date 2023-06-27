import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval3d
from sympy import symbols, simplify, Poly
from sympy.abc import x, y, z,i,j
from code_def import *
from adt import *


def get_enum_tensor(code, indices):
    rank = len(indices)
    stab_group = stabilizer_group(code)
    max_degree = next(iter(stab_group)).length - rank
    pauli_map = {'I':0, "X": 1, "Y":2, "Z":3}
    # enumerator = np.full([4]*rank, fill_value=Polynomial([0]*(max_degree+1)))
    enumerator = [0] if rank==0 else np.full([4]*rank, fill_value=0, dtype = object)
    # print(f"stab length: {len(stab_group)}")
    for stab in stab_group:
        index = [pauli_map[stab.value[i].value] for i in indices]
        weight = stab.weight() - rank + index.count(0)
        wx = stab.W("X") - index.count(1)
        wy = stab.W("Y") - index.count(2)
        wz = stab.W("Z") - index.count(3)
        term = 1 if wx+wy+wz==0 else x**(wx )* y**wy * z**wz
        index = 0 if rank == 0 else tuple(index)
        enumerator[index] += term
    
    return enumerator

def get_BPoly(APoly, n, k):
    lx = symbols('lx')
    poly_coeff = simp_poly(APoly).all_coeffs()[-1::-1]
    poly = 0
    for i in range(len(poly_coeff)):
        lz = (1-lx)/2
        lw = (1+3*lx)/2
        poly += poly_coeff[i]*lz**i*lw**(n-i)
    poly = Poly(simplify(2**k*poly))
    return poly.all_coeffs()[-1::-1]

def distance_from_poly(A_expr, n, k):
    Az_coeff = simp_poly(A_expr).all_coeffs()[-1::-1]
    Bz_coeff = get_BPoly(A_expr, n, k)

    for d in range(len(Az_coeff)):
        if Az_coeff[d] != Bz_coeff[d]:
            return d
        
def AxzNoise(n, APoly, px, wx, pz,  wz):
    return wx**n*wz**n*APoly.subs([(x,px/wx), (z,pz/wz)])

def BxzNoise(n, k, APoly, x, y, z, w):
    return 2**k * AxzNoise(n, APoly, w-z, z+w, (y-x)/2, (x+y)/2)

def simp_poly(enum_poly):
    if isinstance(enum_poly, int):
        return enum_poly
    return Poly(enum_poly.subs([(y,x),(z,x)]))

def xzNoise(n, k, APoly, px, pz):
    APoly = APoly.subs(y,x*z)
    A = AxzNoise(n, APoly, px, 1-px, pz, 1-pz)
    B = BxzNoise(n, k, APoly, px, 1-px, pz, 1-pz)
    # print(f"A: {A}, B: {B}")
    return B-A

def show2Dsimp(enumerator):
    for row in enumerator:
        for col in row:
            print(simp_poly(col))
        print("")

def parse(program):
    insList = program.insList
    tnList = program.tensorList
    tnEnum = get_enum_tensor(tnList[0].tensor, tnList[0].tracted)
    def getMIndex(traceIndex):
        return sum([len(t.tracted) for t in tnList[:traceIndex]])
    for ins in insList:
        if ins[0] == "trace":
            traceIndex, traceLeg, newOneIndex, newOneleg = ins[1:]
            matrixIndex = getMIndex(traceIndex)
            newOne = tnList[newOneIndex]
            newTensor = get_enum_tensor(newOne.tensor, newOne.tracted)
            newTNindex = newOne.tracted.index(newOneleg)    
            tmp = np.tensordot(tnEnum, newTensor, axes= [matrixIndex, newTNindex])
            tnEnum = tmp
            newOne.tracted.pop(0)
            tnList[traceIndex].tracted.pop(0)

        if ins[0] == "self":
            index1, leg1, index2, leg2 = ins[1:]
            mIndex1 = getMIndex(index1)
            mIndex2 = getMIndex(index2)
            tnEnum = np.trace(tnEnum, axis1= mIndex1, axis2= mIndex2)
            tnList[index1].tracted.pop(0)
            tnList[index2].tracted.pop(0)
    return tnEnum[0]        

def eval_code(code, k, px = 0.01, pz = 0.05):
    n = code.length
    APoly = get_enum_tensor(code, [])[0]
    print(simp_poly(APoly),n,k)
    return distance_from_poly(simp_poly(APoly),n,k), xzNoise(n,k,APoly, px, pz)



if __name__ == "__main__":
    n = 5
    px = 0.01
    pz = 0.05
    d,error = eval_code(code11_1_5, 1)
    print(f"d: {d}, error: {error}")


    # tn = TNNetwork(Tensor(code603))
    # tn.trace(0, 0, Tensor(code603), 0)
    # tn.selfTrace(0,1,1,1)
    # tn.setLogical(0,2)

    # exit(0)
    # tn = TNNetwork(Tensor(code603, 6))
    # tn.trace(0, 2, Tensor(code603, 6), 0)
    # tn.trace(0, 3, Tensor(code603, 6), 0)
    # tn.trace(0, 4, Tensor(code603, 6), 0)
    # tn.trace(0, 5, Tensor(code603, 6), 0)
    # tn.setLogical(0,0)

    # n = tn.get_n()
    # k = tn.get_k()
    # print(n,k)
    # APoly = parse(tn)

    
    # print(APoly)
    # print(simp_poly(APoly))
    # print("d", distance_from_poly(simp_poly(APoly), n, k))
    # print(xzNoise(n, k, APoly, px, pz))   
    # exit(0)

    # enumerator = np.tensordot(get_enum_tensor(code603, [3,0,1]) , get_enum_tensor(code603, [0,1]),axes = (2,0))
    # APoly = np.trace(enumerator, axis1=1, axis2=2)[0]
    # print(APoly)
    # print(simp_poly(APoly))
    # print("d", distance_from_poly(simp_poly(APoly),7,1))
    # print(xzNoise(7,1,APoly, px, pz))    
    # APoly = Poly(simp_poly(APoly))

    exit(0)
    print(distance_from_poly(APoly, 7, 1))

