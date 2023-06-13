import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval3d

from adt import *

code513 = codelize(["xzzxi", 'ixzzx', 'xixzz', 'zxixz'])
code604 = codelize(["ixzzxi", 'iixzzx', 'ixixzz', 'izxixz'])
code603 = codelize(['iixxxx', 'iizzzz', 'xixxii', 'ixixix', 'zizizi', 'iziizz'])
code804 = codelize(['iiiixxxx', 'iixxiixx', 'ixixixix', 'xxxxxxxx','iiiizzzz', 'iizziizz', 'iziziziz', 'zzzzzzzz'])
code422 = codelize(['xxxx', 'zzzz'])

def get_enum_tensor(code, indices):
    rank = len(indices)
    stab_group = stabilizer_group(code)
    max_degree = next(iter(stab_group)).length - rank
    pauli_map = {'I':0, "X": 1, "Y":2, "Z":3}
    enumerator = np.full([4]*rank, fill_value=Polynomial([0]*(max_degree+1)))
    for stab in stab_group:
        index = [pauli_map[stab.value[i].value] for i in indices]
        weight = stab.weight() - rank + index.count(0)

        enumerator[tuple(index)] += Polynomial([0]*weight + [1])
    
    return enumerator

def get_BPoly(APoly, n, k):
    x = symbols('x')
    poly_coeff = APoly
    poly = 0
    for i in range(len(poly_coeff)):
        z = (1-x)/2
        w = (1+3*x)/2
        poly += poly_coeff[i]*z**i*w**(n-i)
    poly = Poly(simplify(2**k*poly))
    return poly.all_coeffs()[-1::-1]

def distance_from_poly(Az_coeff, n, k):
    Bz_coeff = get_BPoly(Az_coeff, n, k)
    for d in range(len(Az_coeff)):
        if Az_coeff[d] != Bz_coeff[d]:
            return d


if __name__ == "__main__":
    enumerator = get_enum_tensor(code603, [2,0,1]) @ get_enum_tensor(code603, [0,1])

    APoly = np.trace(enumerator, axis1=1, axis2=2)[0].coef
    print(APoly)
    print(get_BPoly(APoly, 7, 1))
    print(distance_from_poly(APoly, 7, 1))

