import numpy as np
from functools import reduce
import copy
import itertools
from sympy import Matrix

debug = False

from sympy import symbols, simplify, Poly

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def buildProg(insList, initial):
    tn = TNNetwork(initial)
    for ins in insList:
        if ins[0] == 'trace':
            tn.trace(ins[1], ins[2], ins[3], ins[4])
        elif ins[0] == "self":
            tn.selfTrace(ins[1], ins[2], ins[3], ins[4])
        elif ins[0] == "setLog":
            tn.setLogical(ins[1], ins[2])
        else:
            raise NameError(f"no ops as {ins[0]}")
    return tn

class codeTN:
    def __init__(self, stabilizers, name = "", symmetry = None) -> None:
        self.stabs = [stabilizer(i) for i in stabilizers]
        self.name = name
        self.length = self.stabs[0].length
        if not symmetry:
            self.symmetry = list(range(self.length))
        else:
            self.symmetry = symmetry
        
    def merge(self):
        stab_list = []
        length = int(len(self.stabs) / 2 )
        for i in range(length):
            stab_list.append(self.stabs[i] * self.stabs[i+length])
        self.stabs = stab_list
    def __getitem__(self, indices):
        return self.stabs[indices]

    def __len__(self):
        return len(self.stabs)

class Tensor:
    def __init__(self, tensor, index = 0) -> None:
        self.tensor = tensor
        self.size = tensor[0].length
        self.tracted = []
        self.name = tensor.name
        self.index = index

class TNNetwork:
    def __init__(self, initTensor) -> None:
        self.tensorList = [initTensor]
        self.insList = []
        self.Logical = []
        self.traceCount = 0
        self.selfTraceCount = 0

    def trace(self, localIndex, leg1, tensor, leg2):
        if localIndex >= len(self.tensorList):
            raise ValueError(localIndex, len(self.tensorList))
        local = self.tensorList[localIndex]
        if leg1 in local.tracted or leg1 >= local.size:
            raise ValueError(leg1, local.tracted)
        local.tracted.append(leg1)
        tensor.tracted.append(leg2)
        self.insList.append(["trace", localIndex, leg1, len(self.tensorList), leg2])
        self.tensorList.append(tensor)
        

    def selfTrace(self, index1, leg1, index2, leg2):
        tn1 = self.tensorList[index1]
        tn2 = self.tensorList[index2]
        if leg1 in tn1.tracted or leg2 in tn2.tracted:
            raise ValueError(leg1, leg2, tn1.tracted, tn2.tracted)
        tn1.tracted.append(leg1)
        tn2.tracted.append(leg2)
        self.insList.append(["self", index1, leg1, index2, leg2])
        self.selfTraceCount+=1

    def setLogical(self, index, leg):
        self.tensorList[index].tracted.append(leg)
        self.Logical.append((index,leg))

    def get_n(self):
        return sum([tensor.size - len(tensor.tracted) for tensor in self.tensorList])
    
    def get_k(self):
        return len(self.Logical)
    
    def show(self):
        print(str(self))

    def __str__(self):
        return str(self.insList) + "\n" + str([t.name for t in self.tensorList])+ "\n"

    def equiv_trace_leg(self):
        candidate = []
        for i in range(len(self.tensorList)):
            t = self.tensorList[i]
            if len(t.tracted) >= t.size:
                continue
            tmp = [(i,item) for item in range(t.size) if item not in t.tracted]
            if t.name == 'code604':
                tmp = tmp[0:1]
            candidate += tmp
        return candidate
    
    def largerSelfTrace(self, new_ins):
        for ins in self.insList:
            if ins[0]!="self":
                continue
            if selfTraceALargerB(ins, new_ins):
                return True
        return False
    
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


class pauli:
    def __init__(self, value, sign=1) -> None:
        self.value = value.upper()
        self.sign = sign

    def dot(self, other):
        result = pauli_dot(self.value, other.value)
        self.sign *= result[0]*other.sign
        self.value = result[1]

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

    def Wx(self, tag):
        return self.term().count("X")+self.term().count("Y")

    def Wz(self, tag):
        return self.term().count("Z")+self.term().count("Y")
    
    def W(self, tag):
        return self.term().count(tag)

    def prefix(self):
        return product([a.sign for a in self.value])

    def __eq__(self, other) -> bool:
        return self.prefix() == other.prefix() and self.term() == other.term()

    def __str__(self) -> str:
        return f"{str(self.prefix())}*{self.term()}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))


class check_matrix:
    def __init__(self, stabilizers, label=None, symmetry=None) -> None:
        self.matrix = self.setMatrix(stabilizers)
        self.n = stabilizers[0].length
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
            if other.matrix[1, other.n] != 0:
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
        cnrow = this.setOnlyOne1(this.n)
        cn1row = this.find1(this.n+1, [cnrow])
        cn1row = this.setOnlyOne1(this.n+1)
        this.matrix[0] += this.matrix[1]
        this.matrix[cnrow] += this.matrix[cn1row]
        this.matrix = np.delete(this.matrix, (1, cn1row), axis=0)
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
        LogicOp = self.matrix[0:2, :]
        self.matrix = self.matrix[2:, :]
        self.delete([self.n, 0], axis=1)
        self.n -= 1
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
    print("len",len(stab_group))
    for item in stab_group:
        count[item.weight()] += 1
    return count


def Bz_poly(generator, k, stab_group=None):
    n = generator[0].length
    lx = symbols('lx')
    poly_coeff = Az_poly(generator, stab_group)
    poly = 0
    for i in range(len(poly_coeff)):
        lz = (1-lx)
        lw = (1+3*lx)
        poly += poly_coeff[i]*lz**i*lw**(n-i)
    print(poly)
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
    stab_group = [stabilizer("i"*n)]
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
            stab_group.append(product(sub))
    # print("end", len(stab_group))
    return stab_group


def codelize(stabilizers, name = ""):
    return codeTN(stabilizers, name)


def distance(generator, k ,stab_group = None):
    if not stab_group:
        stab_group = stabilizer_group(generator)
    Az_coeff = Az_poly(generator, stab_group)
    Bz_coeff = Bz_poly(generator, k, stab_group)
    print(Az_coeff, Bz_coeff)
    for d in range(len(Az_coeff)):
        if Az_coeff[d] != Bz_coeff[d]:
            return d


def Azx(stab_group, x, y, z, w):
    n = next(iter(stab_group)).length
    result = 0
    for term in stab_group:
        wx = term.Wx("X")
        wz = term.Wz("Z")
        result+=x**wx*y**(n-wx)*z**wz*w**(n-wz)
    return result

def Bzx(k, stab_group, x, y, z, w):
    return 2**k * Azx(stab_group, w-z, z+w, (y-x)/2, (x+y)/2)

def ABzx(stab_group, x,y,z,w,k):
    n = next(iter(stab_group)).length
    Ax = 0
    Bx = 0
    def Nerror(x,y,z,w,wx,wz):
        return x**wx*y**(n-wx)*z**wz*w**(n-wz)
    for term in stab_group:
        wx = term.Wx("X")
        wz = term.Wz("Z")
        Ax += Nerror(x,y,z,w,wx,wz)
        Bx += Nerror(w-z, z+w, (y-x)/2, (x+y)/2,wx,wz)
    return 2**k*Bx-Ax


if __name__ == "__main__":
    code513 = codelize(["xzzxi", 'ixzzx', 'xixzz', 'zxixz'])
    a = ('self',1,1,0,4)
    b = ('self', 1,1,1,3)
    print(selfTraceALargerB(a,b))
