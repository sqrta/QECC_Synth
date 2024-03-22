import numpy as np
from sympy import Matrix
from numpy.linalg import matrix_power as mp
import galois
import os
from interactive import *

import re

def replacenth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    newString = before + after
    return newString

GF = galois.GF(2)

int_range = 2
Terminal = ['l', 'm'] + [str(i) for i in range(1, int_range+1)]
Non_terminal = ['(S+S)', '(S-S)', '(S*S)', 'min(S,S)', 'max(S,S)', 'S**S']

class Statement:
    def __init__(self, init, depth) -> None:
        self.statement = init
        self.depth = depth
    
    def expand(self):
        result = []
        for i in range(1, self.depth+1):
            for NT in Non_terminal:
                newOne = replacenth(self.statement, 'S', NT, i)
                result.append(Statement(newOne, self.depth+1))
        return result

    def terminate(self):
        result = []
        for T in Terminal:
            newOne = self.statement.replace('S', T, 1)
            result.append(Statement(newOne, self.depth-1))
        return result
    
def get_candidate_state(MaxDepth):
    NT_list = [Statement('S', 1)]
    for i in range(MaxDepth):
        newList = []
        for statement in NT_list:
            newList += statement.expand()
        NT_list = newList
    result = []
    while len(NT_list)>0:
        statement = NT_list.pop(0)
        if statement.depth == 0:
            result.append(statement.statement)
        else:
            NT_list += statement.terminate()
    return result
        

def get_S(l):
    M = GF(np.zeros((l,l), dtype=int))
    for i in range(l):
        M[i, (i + 1) % l] = 1
    return M

def get_I(l):
    return GF(np.identity(l, dtype=int))

def get_x(l, m):
    return np.kron(get_S(l), get_I(m))

def get_y(l, m):
    return np.kron(get_I(l), get_S(m))

def ker(M):
    return min(M.shape) - rank(M)

def rank(M):
    return np.linalg.matrix_rank(M)

def writeMatrix(path, M):
    save_dir = './'
    shape = M.shape
    head = "%%MatrixMarket matrix coordinate integer general\n%% Field: GF(2)\n\n"
    with open(save_dir+path, 'w') as f:
        f.write(head)
        f.write(f"{shape[0]} {shape[1]} {2*shape[1]}\n")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if M[i, j]==1:
                    f.write(f"{i+1} {j+1} 1\n")


def Get_kd_BBCode(proc, A, B):
    n = A.shape[0] * 2
    Hz = np.concatenate((B.T, A.T),axis=1)
    Hx = np.concatenate((A, B),axis=1)
    writeMatrix('66Hx.mtx', Hx)
    writeMatrix('66Hz.mtx', Hz)
    rankz = rank(Hz)
    k = n - 2*rankz
    if k==0:
        return 0, 0
    write(proc, 'Read("EvalDis.g");')
    write(proc, 'd;')
    d = int(read(proc))
    return k,d


def search_BBcode(l, m):
    gap = start(['/mnt/e/github/tmp/gap-4.13.0/gap', '-L', 'workplace','-q', '-b'])
    power = get_candidate_state(1)
    power = [str(i) for i in range(13)]
    terms = [(0, i) for i in power] + [(1, i) for i in power]
    n = 2 * l * m
    x = get_x(l, m)
    y = get_y(l, m)
    result = []
    # r1,r2,r3,s1,s2,s3 = 0, 2, 3, 1, 19, 3
    r1,r2,r3,s1,s2,s3 = 0, 1, 15, 2, 23, 25
    def same(p1, p2):
        if p1[0]==p2[0] and eval(p1[1]) == eval(p2[1]):
            return True
        return False
    def mat(p):
        M = x if p[0]==0 else y
        return mp(M, eval(p[1])) 
    
    def valid(p):
        if eval(p)<0 or eval(p)>15:
            return True
        return False
    term_len = len(terms)
    count = 0
    for i1 in range(r1, term_len-2):
        a1 = terms[i1]
        if valid(a1[1]):
            continue
        r2 = i1+1 if i1>r1 else r2
        for i2 in range(r2, term_len-1):
            a2 = terms[i2]
            if valid(a2[1]):
                continue
            if same(a1, a2):
                continue
            r3 = i2+1 if i2>r2 else r3
            for i3 in range(r3, term_len):
                a3 = terms[i3]
                if valid(a3[1]):
                    continue
                if same(a1, a3) or same(a2, a3):
                    continue
                s1 = 0 if i3>r3 else s1
                for j1 in range(s1, term_len-2):
                    b1 = terms[j1]
                    if valid(b1[1]):
                        continue
                    s2 = j1+1 if j1>s1 else s2
                    for j2 in range(s2, term_len-1):
                        b2 = terms[j2]
                        if valid(b2[1]):
                            continue
                        if same(b1, b2):
                            continue
                        s3 = j2+1 if j2>s2 else s3
                        for j3 in range(s3, term_len):
                            b3 = terms[j3]
                            if valid(b3[1]):
                                continue
                            if same(b1, b3) or same(b2, b3):
                                continue
                            print(f"{i1}, {i2}, {i3}, {j1}, {j2}, {j3}")
                            print((a1,a2,a3,b1,b2,b3))
                            A = mat(a1) + mat(a2) + mat(a3)
                            B = mat(b1) + mat(b2) + mat(b3)
                            k, d = Get_kd_BBCode(gap, A, B)
                            if k!=0:
                                count+=1
                            if count>120:
                                terminate(gap)
                                gap = start(['/mnt/e/github/tmp/gap-4.13.0/gap', '-L', 'workplace','-q', '-b'])      
                                count=0                          
                            print(f"good with n: {n}, k: {k}, d: {d}")
                            r_frac = k/(2.0*n) 
                            if (d>3 and r_frac>1/12.0) or (d>2 and r_frac>1/4.0):
                                result.append((a1,a2,a3,b1,b2,b3))
                                print(f"good with n: {n}, k: {k}, d: {d}")
                                print((a1,a2,a3,b1,b2,b3))
                                with open('good_log', 'w+') as f:
                                    f.write((f"good with n: {n}, k: {k}, d: {d}"))
                                    f.write(str((a1,a2,a3,b1,b2,b3)))
                                result.append((a1,a2,a3,b1,b2,b3))
    terminate(gap)
    return result

if __name__ == '__main__':

    
    l = 6
    m = 6
    n = 2*l*m
    
    # gap = start(['/mnt/e/github/tmp/gap-4.13.0/gap', '-L', 'workplace','-q', '-b'])
    # (a1,a2,a3,b1,b2,b3) = ((0, '(l+l)'), (0, '(l+1)'), (0, '(l+2)'), (0, '(l+1)'), (0, '(m-1)'), (1, '2**2'))
    # x = get_x(l, m)
    # y = get_y(l, m)
    # def mat(p):
    #     M = x if p[0]==0 else y
    #     return mp(M, eval(p[1])) 
    # A = mat(a1) + mat(a2) + mat(a3)
    # B = mat(b1) + mat(b2) + mat(b3)
    # for i in range(200):
    #     print(i)
    #     if i==124:

    #         terminate(gap)
    #         gap = start(['/mnt/e/github/tmp/gap-4.13.0/gap', '-L', 'workplace','-q', '-b'])
    #     k, d = Get_kd_BBCode(gap, A, B)
    # print(f"good with n: {n}, k: {k}, d: {d}")
    
    res = search_BBcode(l, m)
    print(res)

    # result = get_candidate_state(1)
    # print(result)
    # print(len(result))
    # print([eval(t) for t in result])
    
    # rankz = rank(Hz)
    # rankx = rank(Hx)
    # length = Hz.T.left_null_space()
    # print(f"Hz shape: {Hz.shape}, rank: {rankx}, {rankz}, length: {length.shape}, k: {n - rankx - rankz}")
    # A_null = ker(A)
    # B_null = ker(B)
    # print("A Space")
    # for vec in A_null:
    #     print(vec.T)
    # print("B Space")
    # for vec in B_null:
    #     print(vec.T)