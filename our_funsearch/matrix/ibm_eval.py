import numpy as np
from sympy import Matrix
from numpy.linalg import matrix_power as mp
import galois
import os
import sys
from interactive import *
from itertools import combinations
from functools import reduce
from gap_path import gap_path

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
    if m==0:
        return get_S(l)
    if l==0:
        return get_I(m)
    return np.kron(get_S(l), get_I(m))

def get_y(l, m):
    if m==0:
        return get_I(l)
    if l==0:
        return get_S(m)
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

def Get_kd_BBCode(proc, A, B, l, m):
    n = A.shape[0] * 2
    Hz = np.concatenate((B.T, A.T),axis=1)
    Hx = np.concatenate((A, B),axis=1)
    writeMatrix(f'{l}{m}Hx.mtx', Hx)
    writeMatrix(f'{l}{m}Hz.mtx', Hz)
    rankz = rank(Hz)
    k = n - 2*rankz
    if k==0:
        return 0, 0
    write(proc, f'lisX:=ReadMTXE("{l}{m}Hx.mtx",0);;')
    write(proc, f'lisZ:=ReadMTXE("{l}{m}Hz.mtx",0);;')
    write(proc, 'Read("EvalDis.g");')
    write(proc, 'd;')
    d = int(read(proc))
    return k,d

def PowStr(p):
    matrix = 'x' if p[0]==0 else 'y'
    return f'{matrix}{p[1]}'

def sortFile(filename, key=('k','d'), func=lambda k,d:k*d+d/100.0):
    with open(filename, 'r') as f:
        content = f.readlines()
        result = []
        for item in content:
            kstring = re.findall(rf"{key[0]}: [0-9]+,", item)[0]
            k = int(re.findall(r"[0-9]+", kstring)[0])
            kstring = re.findall(rf"{key[1]}: [0-9]+,", item)[0]
            d = float(re.findall(r"[0-9]+", kstring)[0])
            result.append((func(k,d), item))
        result.sort(key=lambda a : a[0], reverse=True)
    with open(filename, 'w') as f:
        for item in result:
            f.write(item[1])

def combin(stuff, count):
    return list(combinations(stuff, count))

def good(n,k,d):
    r_frac = k*1.0/(2.0*n)
    if ((d>(n/16.0) and r_frac>=3.0/n) or (d>(n/28.0) and r_frac>4.0/n) or d>8 or (d>3 and k>20)):
        return True
    else:
        return False

def sumMat(mList):
    return reduce(lambda a,b : a+b, mList)

def pruneTerm(Terms):
    result = []
    for term in Terms:
        sumT = sum([a[0] for a in term])
        if sumT % len(term) == 0:
            if term[0][1]=='0':
                result.append(term)
        else:
            result.append(term)
    return result
    
def search_2GBAcode(l, m, countA, countB, kthres = 4, init=0):
    filename = f'good_log_{l}_{m}_count{countA}{countB}'
    ri, rj = init,0
    if ri==0 and rj==0:
        with open(filename, 'w') as f:
            pass    
    else:
        with open(filename, 'a') as f:
            pass
    gap = start([gap_path, '-L', 'workplace','-q', '-b'])
    power = get_candidate_state(1)
    power = [str(i) for i in range(6)]
    terms = [(0, str(i)) for i in range(l)] + [(1, str(i)) for i in range(1, m)]
    n = 2 * l * m
    x = get_x(l, m)
    y = get_y(l, m)
    result = []
    
    found = set()

    def mat(p):
        M = x if p[0]==0 else y
        res = mp(M, eval(p[1])) 
        return res
    
    def goodC(n,k,d):
        r = d*k*1.0/n
        if r > 0.5 or (k>n/4 and d>2) or (d>n/6 and k>5):
            return True
        return False
    
    term_len = len(terms)
    count = 0
    Aterms = combin(terms, countA)
    Aterms = pruneTerm(Aterms)
    # print(Aterms)
    # for i in range(len(Aterms)):
    #     aterm = Aterms[i]
    #     print(f"{i}, {'+'.join([PowStr(a1) for a1 in aterm])}")
    # exit(0)
    iter_count = 0
    for i in range(ri, len(Aterms)):
        aterm = Aterms[i]
        print(f"i: {i}, iter: {iter_count},{'+'.join([PowStr(a1) for a1 in aterm])}")
        A = sumMat([mat(t) for t in aterm])
        rankA = rank(A)
        if rankA>=l*m-kthres:
            print([PowStr(a1) for a1 in aterm], f"lm-rankA: {l*m-rankA}")

            continue
        if countA == countB:
            Bterms = Aterms[i:]
        else:
            Bterms = combin(terms, countB)
        jstart = rj if i==ri else 0
        for j in range(jstart, len(Bterms)):
            iter_count += 1
            bterm = Bterms[j]
            B = sumMat([mat(t) for t in bterm])
            k, d = Get_kd_BBCode(gap, A, B, l, m)
            r = d*k*1.0/n   
            # print(f"i: {i}, j: {j} with n: {n}, k: {k}, d: {d}, r: {r}") 
            # print(f"{iter_count},{'+'.join([PowStr(a1) for a1 in aterm])}, {'+'.join([PowStr(a1) for a1 in bterm])}")
            if True and goodC(n,k,d):
                # found.add((n,k,d))
                with open(filename, 'a') as f:
                    f.write((f"i:{i}, good with n: {n}, k: {k}, d: {d}, r: {r} "))
                    f.write(f"{'+'.join([PowStr(a1) for a1 in aterm])}, {'+'.join([PowStr(a1) for a1 in bterm])}, {l}, {m}\n")
            if k!=0:
                count+=1
            if count>120:
                count = 1
                terminate(gap)
                gap = start([gap_path, '-L', 'workplace','-q', '-b'])      

            
    print(f"itercount: {iter_count}")
    terminate(gap)
    sortFile(filename)
    with open('finished.txt', 'a') as f:
        f.write(f"{(l,m)}\n")
    if count>0:
        os.system(f'rm {l}{m}Hx.mtx')
        os.system(f'rm {l}{m}Hz.mtx')
    return result

if __name__ == '__main__':

    def mat(p):
        M = r if p[0]=='x' else s
        res = mp(M, int(p[1:])) 
        return res


    toSearch = [] 
    countA = 4
    countB = 4
    flag = sys.argv[1]

    import time
    startT = time.time()
    for i in range(2, len(sys.argv)):
        lmstring = sys.argv[i].split(',')
        l, m = int(lmstring[0]), int(lmstring[1])
        init = 0
        if len(lmstring)>2:
            init = int(lmstring[2])
        toSearch.append((l,m,init))
    with open('isWorking.txt', 'a') as f:
        f.write(f"{toSearch}\n")
    if flag == 'sort':
        path = 'BB_sopt2/'
        for item in toSearch:
            l,m = item
            filename = path + f'good_log_{l}_{m}_count33'
            sortFile(filename)
    if flag == 'test':
        gap = start([gap_path, '-L', 'workplace','-q', '-b'])
        l = 4
        m = 30
        # # x^9 + y^3 + y^6$ & $1+x^2 +x^7
        # aterm = ((0, '1'), (0, '2'), (1, '3'))
        # bterm = ((0, '3'), (1, '5'), (1, '10'))
        with open('toRun.txt', 'r') as f:
            toRuns = f.readlines()
        for t in toRuns:
            terms = t.split(', ')
            l = int(terms[5])
            m = int(terms[6])
            aterm = terms[3].split(' ')[2].split('+')
            bterm = terms[4].split('+')
            print(l,m,aterm,bterm)
            r = get_x(l, m)
            # print(r)
            s = get_y(l, m)
            # A = mp(r, 0) + mp(r, 1) @ mp(s, 4)
            # B = mp(r, 0) + mp(r, 1) + mp(r, 2) + mp(s, 1) + mp(s, 3) @ mp(r, 1) + mp(s, 2) @ mp(r, 6) 
            A = sumMat([mat(t) for t in aterm])
            B = sumMat([mat(t) for t in bterm])

            k,d = Get_kd_BBCode(gap, A, B, l, m)
            print(k,d)
        os.system(f'rm *.mtx')
        exit(0)
    if flag == 'bb':
        for item in toSearch:
            l,m = item
            n = 2*l*m
            res = search_BBcode(l, m)
            print(f"{l},{m} finish")
    if flag[:2] == '2g':
        counts = flag.split('_')
        countA, countB, kthres = int(counts[1]), int(counts[2]), int(counts[3])
        for item in toSearch:
            l,m,init = item
            n = 2*l*m
            res = search_2GBAcode(l, m, countA, countB, kthres,init)
            print(f"{l},{m},{countA}, {countB} finish")
    endT = time.time()
    print(toSearch)
    print(f"use {endT-startT}s")

    # with open(f'good_log_{l}_{m}', 'w') as f:
    #     for i in res:
    #         n,k,d,a1,a2,a3,b1,b2,b3 = i
    #         r_frac = k*1.0/(2.0*n)
    #         f.write((f"good with n: {n}, k: {k}, d: {d}, r: {r_frac} "))
    #         f.write(f"{PowStr(a1)}+{PowStr(a2)}+{PowStr(a3)}, {PowStr(b1)}+{PowStr(b2)}+{PowStr(b3)}\n")

    # gap = start([gap_path, '-L', 'workplace','-q', '-b'])
    # l = 15
    # m = 3
    # x = get_x(l, m)
    # y = get_y(l, m)
    # A = mp(x, 9) + mp(y, 1) + mp(y, 2)
    # B = mp(y, 0) + mp(x, 2) + mp(x, 7)
    # k, d = Get_kd_BBCode(gap, A, B)
    # print(k,d)
    
    

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

# good with n: 72, k: 16, d: 4((0, '1'), (0, '0'), (0, '2'), (0, '3'), (1, '2'), (1, '4'))