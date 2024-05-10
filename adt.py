import numpy as np
from functools import reduce
from numpy import kron
import functools
import copy
import itertools
from sympy import Matrix

import psutil
import os

pid = os.getpid()
process = psutil.Process(pid)

debug = False

def get_mem_use(mode='MB'):
    mem_info = process.memory_info()
    if mode == 'KB':
        mem_usage =  mem_info.rss / 1024.0
    elif mode == 'MB':
        mem_usage = mem_info.rss / 1024.0 / 1024.0  # 单位转换为MB
    else:
        mem_usage =  mem_info.rss
    return mem_usage

def show_mem_use(mode='MB', msg=''):
    mem_usage = get_mem_use(mode)
    print(f"{msg}, memory usage {mem_usage:.2f} {mode}") 

debug = False


def tensor(op_list):
    return functools.reduce(kron, op_list, 1)
from sympy import symbols, simplify, Poly

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.array([[1, 0], [0, 1]], dtype=complex)
Pauli2MatrixDict = {'I':I, 'X':X, 'Y':Y, 'Z':Z}
Int2Matrix = [I,X,Y,Z]
    
def selfTraceALargerB(a, b):
    for i in range(1,5):
        if a[i]>b[i]:
            return True
        elif a[i]<b[i]:
            return False
    return False


class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def numberToBase(n, b, length):
    if n == 0:
        return [0 for i in range(length)]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return [0 for i in range(length-len(digits))] + digits[::-1]

def intVec2M(vec, n):
    v = [0]*(n-len(vec))+vec
    return tensor([Int2Matrix[a] for a in v])

def getAllPauli(n):
    size = 4**n
    return [intVec2M(numberToBase(i, 4, n)) for i in range(size)]

def getAllPauliInt(n):
    size = 4**n
    return [numberToBase(i, 4, n) for i in range(size)]

def allsubset(s):
    return reduce(lambda a, b: a+b, [findsubsets(s, i) for i in range(len(s)+1)])


def product(terms):
    return reduce(lambda a, b: a*b, terms)


def pauli_dot(a, b):
    if a == "I":
        return (1, b)
    if b == "I":
        return (1, a)
    if a == b:
        return (1, "I")
    rule = {("X", "Z"): (-1j, "Y"), ("Z", "X"): (1j, "Y"), ("X", "Y"): (1, "Z"),
            ("Y", "X"): (-1, "Z"), ("Y", "Z"): (1, "X"), ("Z", "Y"): (-1, "X")}
    result = rule[(a, b)]
    return result


def vec2stabilizer(vec):
    n = int(len(vec)/2)
    xvec = vec[:n]
    zvec = vec[n:]
    a = "".join(['x' if i == 1 else 'i' for i in xvec])
    b = "".join(['z' if i == 1 else 'i' for i in zvec])
    return stabilizer(a)*stabilizer(b)


def checkM2Stabilizers(M):
    return [vec2stabilizer(vec) for vec in M]

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

class pauli:
    def __init__(self, value, sign=1) -> None:
        self.value = value.upper()
        self.sign = sign

    def dot(self, other):
        result = pauli_dot(self.value, other.value)
        self.sign *= result[0]*other.sign
        self.value = result[1]

    def isTag(self, tag):
        if isinstance(tag, str):
            return self.value == tag.upper()
        for t in tag:
            if self.value == t.upper():
                return True
        return False

    def toInt(self):
        pauliMap = {'I':0, 'X':1, 'Y':2, 'Z':3}
        return pauliMap[self.value]
    
    def Matrix(self):
        return Pauli2MatrixDict[self.value]

    def __eq__(self, other) -> bool:
        return self.sign == other.sign and self.value == other.value

    def __str__(self) -> str:
        sign = "" if self.sign == 1 else self.sign
        return sign+self.value

    def __repr__(self) -> str:
        return str(self)


class stabilizer:
    def __init__(self, value) -> None:
        self.length = len(value)
        self.value = [pauli(s) for s in value]

    def __mul__(self, other):
        c = copy.deepcopy(self)
        for i in range(self.length):
            c.value[i].dot(other.value[i])
        return c

    def term(self):
        return "".join([a.value for a in self.value])

    def weight(self):
        return self.length-self.term().count("I")

    def Wx(self):
        return self.term().count("X")+self.term().count("Y")

    def Wz(self):
        return self.term().count("Z")+self.term().count("Y")
    
    def W(self, tag):
        return self.term().count(tag)
    
    def tagVec(self, tag):
        return [p.isTag(tag) for p in self.value]
    
    def toInt(self):
        return [p.toInt() for p in self.value]
    
    def toMatrix(self):
        return tensor([a.Matrix() for a in self.value])

    def prefix(self):
        return product([a.sign for a in self.value])

    def __eq__(self, other) -> bool:
        # return self.prefix() == other.prefix() and self.term() == other.term()
        return self.term() == other.term()

    def __str__(self) -> str:
        return f"{self.term()}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))


class check_matrix:
    def __init__(self, stabilizers, label=None, symmetry=None) -> None:
        self.matrix = self.setMatrix(stabilizers)
        self.n = stabilizers[0].length
        self.LogicOp = []
        if symmetry:
            self.symmetry = symmetry
        else:
            self.symmetry = [[i] for i in range(self.n)]
        if label == None:
            self.label = bidict({i: i for i in range(self.n)})
            # {qubit label : column}
        else:
            self.label = label

    def setMatrix(self, stabilizers):
        n = stabilizers[0].length
        matrix = np.zeros((len(stabilizers), 2*n), dtype=np.int16)
        for i in range(len(stabilizers)):
            stab = stabilizers[i].term()
            for j in range(len(stab)):
                c = stab[j]
                if c == "X":
                    matrix[i, j] = 1
                elif c == 'Z':
                    matrix[i, n + j] = 1
                elif c=="Y":
                    matrix[i, j] = 1
                    matrix[i, n + j] = 1
        return matrix

    def swapColumn(self, a, b):
        self.matrix[:, [a, b]] = self.matrix[:, [b, a]]

    def swapRow(self, a, b):
        self.matrix[[a, b], :] = self.matrix[[b, a], :]

    def swapLeg(self, a, b):
        self.swapColumn(a, b)
        self.swapColumn(a+self.n, b + self.n)
        self.swapLabel(a, b)

    def swapLabel(self, a, b):
        # swap the label of two column
        label1 = self.label.inverse[a][0]
        label2 = self.label.inverse[b][0]
        self.label[label1] = b
        self.label[label2] = a

    def row_echelon(self):
        matrix = Matrix(self.matrix)
        matrix_rref = matrix.rref()
        self.matrix = np.array(matrix_rref[0].tolist(), dtype=np.int16)
        self.Mod2()

    def Mod2(self):
        shape = self.matrix.shape
        M = np.zeros(shape, dtype=np.int16)
        for i in range(shape[0]):
            for j in range(shape[1]):
                M[i, j] = 0 if self.matrix[i, j] % 2 == 0 else 1
        self.matrix = M

    def rowWBound(self):
        stabs = [row[:self.n]+row[self.n:] for row in self.matrix]
        count = [np.count_nonzero(row!=0) for row in stabs]
        if len(count)==0:
            return 0
        return max(count)
    
    def colWBound(self):
        count = self.colW()
        if len(count)==0:
            return 0
        return max(count)
    
    def colW(self):
        stabs = [row[:self.n]+row[self.n:] for row in self.matrix]
        count = [0 for i in range(self.n)]
        for row in stabs:
            for i in range(self.n):
                if row[i]!=0:
                    count[i]+=1
        return count

    def setOnlyOne1(self, column):
        target = -1
        for i in range(self.matrix.shape[0]-1, -1, -1):
            if self.matrix[i, column] == 1:
                target = i
                break
        if target == -1:
            return target
        for i in range(target):
            if self.matrix[i, column] == 1:
                self.matrix[i] += self.matrix[target]
        self.Mod2()
        return target

    def find1(self, column, default=[-1]):
        target = -1
        for i in range(self.matrix.shape[0]-1, -1, -1):
            if self.matrix[i, column] == 1:
                target = i
                if target not in default:
                    break
        return target

    def setTraceForm(self, column):
        index1 = list(range(self.n))
        index1.remove(column)
        index1.insert(0, column)
        index2 = [i+self.n for i in index1]
        index = index1 + index2

        # self.swapColumn(0, column)
        # self.swapColumn(self.n, column+self.n)
        # self.swapLabel(0, column)
        self.matrix = self.matrix[:, index]
        # print("swap")
        # print(self.matrix)
        self.row_echelon()
        target = self.setOnlyOne1(self.n)

        if target > 0:
            if self.matrix[0, 0] != 0:
                self.matrix[[1, target], :] = self.matrix[[target, 1], :]
            else:
                self.matrix[[0, target], :] = self.matrix[[target, 0], :]

    def resetLabel(self):
        self.n = int(self.matrix.shape[1]/2)
        self.label = bidict({i: i for i in range(self.n)})

    def removeZeroRow(self):
        count = ~np.all(self.matrix == 0, axis=1)
        self.matrix = self.matrix[count]  

    def trace(self, other_tensor, label1, label2):
        # column1 = self.label[label1]
        # column2 = other.label[label2]

        column1 = label1
        column2 = label2
        # print("trace", label1, column1, label2, column2)
        this = copy.deepcopy(self)
        other = copy.deepcopy(other_tensor)
        this.symmetry = set_symmetry(self.symmetry, column1)
        this.symmetry += set_symmetry(other.symmetry, column2, shift=self.n-1)
        this.setTraceForm(column1)
        other.setTraceForm(column2)
        # print(this.n, other.n)
        # print(this.matrix)
        # print(other.matrix)
        if this.matrix[1, self.n] == 0:
            print("branch")
            if other.matrix[1, other.n] != 0:

                width = this.n + other.n - 2
                height = this.matrix.shape[0] + other.matrix.shape[0]-2
                M = np.zeros((height, 2 * width), dtype=np.int16)
                M[0, 0:this.n-1] = this.matrix[0, 1: this.n]
                M[0, this.n-1:width] = this.matrix[0, 0]*other.matrix[0,
                                                                      1:other.n] + this.matrix[0, this.n] * other.matrix[1, 1:other.n]
                M[0, width:width+this.n-1] = this.matrix[0, this.n+1:]
                M[0, width+this.n-1:] = this.matrix[0, 0] * other.matrix[0,
                                                                         other.n+1:] + this.matrix[0, this.n]*other.matrix[1, other.n+1:]
                M[1:this.matrix.shape[0], 0:this.n-1] = this.matrix[1:, 1: this.n]
                M[1:this.matrix.shape[0], width:width +
                    this.n-1] = this.matrix[1:, this.n+1:]
                M[this.matrix.shape[0]:, this.n -
                    1:width] = other.matrix[2:, 1:other.n]
                M[this.matrix.shape[0]:, width+this.n -
                    1:] = other.matrix[2:, other.n+1:]
                print(M)
            else:
                width = this.n + other.n - 2
                height = this.matrix.shape[0] + other.matrix.shape[0]-1
                M = np.zeros((height, 2 * width), dtype=np.int16)
                f = 1 if this.matrix[0, 0] == other.matrix[0,
                                                           0] and this.matrix[0, this.n] == other.matrix[0, other.n] else 0
                M[0, 0:this.n-1] = f*this.matrix[0, 1: this.n]
                M[0, this.n-1:width] = f*other.matrix[0, 1:other.n]
                M[0, width:width+this.n-1] = f*this.matrix[0, this.n+1:]
                M[0, width+this.n-1:] = f * other.matrix[0, other.n+1:]
                M[1:this.matrix.shape[0], 0:this.n -
                    1] = this.matrix[1:this.matrix.shape[0], 1: this.n]
                M[1:this.matrix.shape[0], width:width+this.n -
                    1] = this.matrix[1:this.matrix.shape[0], this.n+1:]
                M[this.matrix.shape[0]:, this.n -
                    1:width] = other.matrix[1:other.matrix.shape[0], 1:other.n]
                M[this.matrix.shape[0]:, width+this.n -
                    1:] = other.matrix[1:other.matrix.shape[0], other.n+1:]
        else:
            if other.matrix.shape[0]>1 and other.matrix[1, other.n] != 0:
                # print("trace")
                width = this.n + other.n - 2
                height = this.matrix.shape[0] + other.matrix.shape[0] - 2
                M = np.zeros((height, 2 * width), dtype=np.int16)
                M[0:2, 0:this.n-1] = this.matrix[0:2, 1: this.n]
                M[0:2, this.n-1:width] = other.matrix[0:2, 1:other.n]
                M[0:2, width:width+this.n-1] = this.matrix[0:2, this.n+1:]
                M[0:2, width+this.n-1:] = other.matrix[0:2, other.n+1:]
                M[2:this.matrix.shape[0], 0:this.n-1] = this.matrix[2:, 1: this.n]
                M[2:this.matrix.shape[0], width:width +
                    this.n-1] = this.matrix[2:, this.n+1:]
                M[this.matrix.shape[0]:, this.n -
                    1:width] = other.matrix[2:, 1:other.n]
                M[this.matrix.shape[0]:, width+this.n -
                    1:] = other.matrix[2:, other.n+1:]
            else:
                print("2branch")
                i = other.matrix[0, 0]
                j = other.matrix[0, other.n]
                width = this.n + other.n - 2
                height = this.matrix.shape[0] + other.matrix.shape[0]-2
                M = np.zeros((height, 2 * width), dtype=np.int16)
                M[0, 0:this.n-1] = i*this.matrix[0, 1: this.n] - \
                    j*this.matrix[1, 1: this.n]
                M[0, this.n-1:width] = other.matrix[0, 1:other.n]
                M[0, width:width+this.n-1] = i*this.matrix[0,
                                                           this.n+1:] - j * this.matrix[1, this.n+1:]
                M[0, width+this.n-1:] = other.matrix[0, other.n+1:]
                M[1:this.matrix.shape[0]-1, 0:this.n -
                    1] = this.matrix[2:this.matrix.shape[0], 1: this.n]
                M[1:this.matrix.shape[0]-1, width:width+this.n -
                    1] = this.matrix[2:this.matrix.shape[0], this.n+1:]
                M[this.matrix.shape[0]-1:, this.n -
                    1:width] = other.matrix[1:other.matrix.shape[0], 1:other.n]
                M[this.matrix.shape[0]-1:, width+this.n -
                    1:] = other.matrix[1:other.matrix.shape[0], other.n+1:]
        this.matrix = M
        this.Mod2()
        this.removeZeroRow()
        this.n = this.n + other.n - 2
        this.resetLabel()
        return this

    def selfTrace(self, label1, label2):
        # column1 = self.label[label1]
        # column2 = self.label[label2]
        # print("self", label1, column1, label2, column2)

        column1 = label1
        column2 = label2
        self.symmetry = set_symmetry(self.symmetry, max(column1, column2))
        self.symmetry = set_symmetry(self.symmetry, min(column1, column2))
        index1 = list(range(self.n))
        index1.remove(column1)
        index1.remove(column2)
        index1.insert(0, column2)
        index1.insert(0, column1)
        index2 = [i+self.n for i in index1]
        index = index1 + index2
        this = copy.deepcopy(self)
        this.matrix = this.matrix[:, index]
        this.row_echelon()
        c1row = this.setOnlyOne1(1)
        this.swapRow(c1row, 1)
        cnrow = this.setOnlyOne1(this.n)
        this.swapRow(cnrow,2)
        cnrow=2

        cn1row = this.find1(this.n+1, [0,1,2])
        cn1row = this.setOnlyOne1(this.n+1)
        this.matrix[0] += this.matrix[1]
        this.matrix[cnrow] += this.matrix[cn1row]
        if c1row == 0:
            todel = (cn1row,)
        else:
            todel = (1, cn1row)
        this.matrix = np.delete(this.matrix, todel, axis=0)
        this.matrix = np.delete(this.matrix, (0, 1, this.n, this.n+1), axis=1)
        this.n -= 2
        this.Mod2()
        return this

    def delete(self, rows, axis=0):
        self.matrix = np.delete(self.matrix, rows, axis=axis)

    def move2firstCol(self, column):
        index1 = list(range(self.n))
        index1.remove(column)
        index1.insert(0, column)
        index2 = [i+self.n for i in index1]
        index = index1 + index2
        self.matrix = self.matrix[:, index]

    def setLogical(self, label):
        column = label
        self.move2firstCol(column)
        self.row_echelon()
        row = self.setOnlyOne1(self.n)
        self.swapRow(row, 1)
        cond = (self.matrix[:,0]!=0) | (self.matrix[:,self.n]!=0)
        LogicOp = self.matrix[cond]
        self.matrix = self.matrix[~cond]

        self.delete([self.n, 0], axis=1)
        self.n -= 1
        self.LogicOp.append(LogicOp) 
        return LogicOp

def set_symmetry(groups, remove, shift=0):
    symmetry = []
    for group in groups:
        tmp = []
        for item in group:
            if item < remove:
                tmp.append(item+shift)
            elif item > remove:
                tmp.append(item-1+shift)
        if len(tmp) > 0:
            symmetry.append(tmp)
    return symmetry


def Az_poly(generator, stab_group=None):
    n = generator[0].length
    if not stab_group:
        stab_group = stabilizer_group(generator)
    count = [0] * (n+1)
    for item in stab_group:

        count[item.weight()] += 1
    result = []
    norm = count[0]
    return [i//norm for i in count]

def Az_poly2(generator, stab_group=None):
    n = generator[0].length
    if not stab_group:
        stab_group = stabilizer_group(generator)
    count = [0] * (n+1)
    for item in stab_group:
        count[item.Wx()+item.Wz()] += 1
    result = []
    norm = count[0]
    return [i//norm for i in count]


def Bz_poly(generator, k, stab_group=None):
    n = generator[0].length
    lx = symbols('lx')
    poly_coeff = Az_poly(generator, stab_group)
    poly = 0
    for i in range(len(poly_coeff)):
        lz = (1-lx)
        lw = (1+3*lx)
        poly += poly_coeff[i]*lz**i*lw**(n-i)
    poly = Poly(simplify(2**k*poly))
    coeffs = poly.all_coeffs()[-1::-1]
    coeffs = [i//coeffs[0] for i in coeffs]
    return coeffs


def Az(w, z, generator, k):
    n = generator[0].length
    count = Az_poly(generator)
    result = 0
    for i in range(n):
        result += count[i]*(z/w)**i
    return w**n*result


def Bz(w, z, generator, k):
    return 4**k * Az((1+3*z)/2, (1-z)/2, generator, k) / 2**k


def stabilizer_group(generator):
    subsets = allsubset(generator)
    
    n = generator[0].length
    stab_group = set()
    stab_group.add(stabilizer("i"*n))
    length = len(subsets)
    if debug:
        print(f"subset lenght: {len(subsets)}")
        print(f"length: {length}")
    percent = max(length // 10, 1)
    for i in range(length):
        sub = subsets[i]

        if i%percent == 0 and debug:
            print(f"{10 * i//percent}%")
        if len(sub) > 0:           
            stab_group.add(product(sub))
    # print("end", len(stab_group))
    return stab_group


def codelize(stabilizers, name = ""):
    return codeTN(stabilizers, name)


def distance(generator, k ,stab_group = None):
    if not stab_group:
        stab_group = stabilizer_group(generator)
    Az_coeff = Az_poly(generator, stab_group)
    Bz_coeff = Bz_poly(generator, k, stab_group)
    # print(Az_coeff, Bz_coeff)
    for d in range(len(Az_coeff)):
        if Az_coeff[d] != Bz_coeff[d]:
            return d


def Azx(stab_group, x, y, z, w):
    n = next(iter(stab_group)).length
    result = 0
    for term in stab_group:
        wx = term.Wx()
        wz = term.Wz()
        result+=x**wx*y**(n-wx)*z**wz*w**(n-wz)
    return result

def Bzx(k, stab_group, x, y, z, w):
    return 2**k * Azx(stab_group, w-z, z+w, (y-x)/2, (x+y)/2)

def ABzx(stab_group, x,y,z,w,k,K=1):
    n = next(iter(stab_group)).length
    Ax = 0
    Bx = 0
    def Nerror(x,y,z,w,wx,wz):
        return x**wx*y**(n-wx)*z**wz*w**(n-wz)
    for term in stab_group:
        wx = term.Wx()
        wz = term.Wz()
        Ax += Nerror(x,y,z,w,wx,wz)
        Bx += Nerror(w-z, z+w, (y-x)/2, (x+y)/2,wx,wz)
    # print(Ax, 2**k*Bx/K)
    res = 2**k/K*Bx-Ax
    return 2**k/K*Bx-Ax, 1- Ax/(2**k/K*Bx)

def ABzxVec(stab_group, x,xp,z,zp,k,K=1):
    n = next(iter(stab_group)).length
    Ax = 0
    Bx = 0
    def Nerror(x,xp,z,zp):
        return product(x)*product(xp)*product(z)*product(zp)
    def errorVec(errorVec, qubitmap):
        error = []
        for i in range(qubitmap):
            if qubitmap[i]:
                error.append(errorVec)
        return error
    for term in stab_group:
        xmap = term.tagVec(('x', 'y'))
        zmap = term.tagVec(('z', 'y'))
        xe = errorVec(x, xmap)
        ze = errorVec(z, zmap)
        xpe = errorVec(xp, xmap)
        zpe = errorVec(zp, zmap)
        Ax += Nerror(xe, xpe, )

def prog2Cm(insList, tnList):
    
    tracted = [[] for i in range(len(tnList))]
    cm = check_matrix(tnList[0])
    def getMIndex(traceIndex, traceLeg):
        index = 0
        for i in range(traceIndex):
            index+=tnList[i].length - len(tracted[i])
        count = 0
        for tractedLeg in tracted[traceIndex]:
            if tractedLeg<traceLeg:
                count+=1
        index += traceLeg - count
        return index
    
    for ins in insList:
        if ins[0]=="trace":
            traceIndex, traceLeg, newOneIndex, newOneleg = ins[1:]
            matrixIndex = getMIndex(traceIndex, traceLeg)
            newOne = tnList[newOneIndex] 
            cm = cm.trace(check_matrix(newOne), matrixIndex, newOneleg)
            # print(matrixIndex, newOneleg)
            tracted[traceIndex].append(traceLeg)
            tracted[newOneIndex].append(newOneleg)
        elif ins[0] == "self":
            index1, leg1, index2, leg2 = ins[1:]
            mIndex1 = getMIndex(index1, leg1)
            mIndex2 = getMIndex(index2, leg2)
            cm = cm.selfTrace(mIndex1, mIndex2)
            # print(mIndex1, mIndex2)
            tracted[index1].append(leg1)
            tracted[index2].append(leg2)
        elif ins[0] == "setLog":
            index1, leg1 = ins[1:]
            mIndex1 = getMIndex(index1, leg1)
            cm.setLogical(mIndex1)
            # print(mIndex1)
            tracted[index1].append(leg1)
        else:
            raise NameError(f"no ops as {ins[0]}")
    cm.removeZeroRow()
    return cm




if __name__ == "__main__":
    code513 = codelize(["xzzxi", 'ixzzx', 'xixzz', 'zxixz'])
    a = ('self',1,1,0,4)
    b = ('self', 1,1,1,3)
    print(selfTraceALargerB(a,b))
