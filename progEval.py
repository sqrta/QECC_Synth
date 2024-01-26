from adt import *
import itertools
from enumerator import *
from statistics import mean,geometric_mean

def TNN2CM(TN):
    tn = copy.deepcopy(TN)
    insList = tn.insList
    tnList = tn.tensorList
    cm = check_matrix(tnList[0].tensor())
    def getMIndex(traceIndex):
        return sum([len(t.tracted) for t in tnList[:traceIndex]])
    for ins in insList:
        if ins[0] == "trace":
            traceIndex, traceLeg, newOneIndex, newOneleg = ins[1:]
            matrixIndex = getMIndex(traceIndex)
            newOne = tnList[newOneIndex]
            newTNindex = newOne.tracted.index(newOneleg)    
            # print('trace', matrixIndex, newTNindex)

            tmp = cm.trace(check_matrix(newOne.tensor()), matrixIndex, newTNindex)
            cm = tmp
            newOne.tracted.pop(0)
            tnList[traceIndex].tracted.pop(0)
        if ins[0] == "self":
            index1, leg1, index2, leg2 = ins[1:]
            mIndex1 = getMIndex(index1)
            mIndex2 = getMIndex(index2)
            print(ins)
            print('self', mIndex1, mIndex2)
            cm = cm.selfTrace(mIndex1, mIndex2)
            tnList[index1].tracted.pop(0)
            tnList[index2].tracted.pop(0)
    return cm

def ErrorFrCode(px, pz, cm):
    rbsize = 7
    colw = [a + rbsize for a in cm.colW()]
    xe = [1-(1-px)**i for i in colw]
    ze = [1-(1-pz)**i for i in colw]
    return xe, ze

def prog2Stab(prog):
    tmp = prog2CM(prog)
    stab = checkM2Stabilizers(tmp.matrix)
    return stab

def prog2CM(prog):
    insList = prog[0]
    tnList = prog[1]
    tn = prog2TNN(insList, tnList)
    cm = tn.toCm()
    return cm

def ProgMetric(prog, px=0.01, pz=0.05):
    insList = prog[0]
    tnList = prog[1]
    tensorList  = [eval(t) for t in tnList]

    tn = prog2TNN(insList, tnList)
    n = tn.get_n()
    k = tn.get_k()

    tmp = tn.toCm()
    tmp.row_echelon()
    rw = tmp.rowWBound()
    cw = tmp.colWBound()
    # tn.setLogical(0,0)
    d,error,K = eval_TN(tn, px, pz)
    return error

def eval_prog(prog, px, pz):
    insList = prog[0]
    tnList = prog[1]
    tensorList  = [eval(t) for t in tnList]
    # a = prog2Cm(insList, tensorList)
    # a.row_echelon()

    tn = prog2TNN(insList, tnList)
    n = tn.get_n()
    k = tn.get_k()

    tmp = tn.toCm()
    tmp.row_echelon()
    rw = tmp.rowWBound()
    cw = tmp.colWBound()
    # tn.setLogical(0,0)

    d,error,K = eval_TN(tn, px, pz)

    print(f"n: {n}, k: {k}, d: {d}, Ks: {K}, rW: {rw}, cW: {cw}, error: {error:.5e}")

    code = checkM2Stabilizers(tmp.matrix)
    print(code)
    print([stab.toInt() for stab in code])
    print(tmp.matrix)
    # for op in tmp.LogicOp:
    #     print(op.shape[1]//2)
    #     print(op)
    stab_group = stabilizer_group(code)

    print(f"ABzx: {ABzx(stab_group, px, 1 - px, pz, 1- pz, k, K)}, d: {distance(code, k, stab_group)}")
    # print(eval_tn(tn))
    

def EvalActiveCode(prog, px, pz):
    insList = prog[0]
    tnList = prog[1]
    tensorList  = [eval(t) for t in tnList]
    # a = prog2Cm(insList, tensorList)
    # a.row_echelon()

    tn = prog2TNN(insList, tnList)
    n = tn.get_n()
    k = tn.get_k()

    CM = tn.toCm()
    CM.row_echelon()
    rw = CM.rowWBound()
    cw = CM.colWBound()
    # tn.setLogical(0,0)
    d,error,K = eval_TN(tn)
    print(f"n: {n}, k: {k}, d: {d}, Ks: {K}, rW: {rw}, cW: {cw}, error: {error:.5e}")
    return evalFromCeckKMatrix(CM, px, pz, k, K)

def evalFromCeckKMatrix(CM, px, pz, k=1, K=1):
    code = checkM2Stabilizers(CM.matrix)
    rbsize = 7
    colw = [a + rbsize for a in CM.colW()]
    xe = geometric_mean([1-(1-px)**i for i in colw])
    ze = geometric_mean([1-(1-pz)**i for i in colw])
    stab_group = stabilizer_group(code)
    print(f"xe: {xe:.4f}, ze: {ze:.4f}")
    error_rate = ABzx(stab_group, xe, 1 - xe, ze, 1- ze, k, K)
    print(f"Error rate: {error_rate: .5e}")
    return error_rate

prog15_14 = ([['trace', 0, 2, 1, 0], ['trace', 0, 4, 2, 0], ['trace', 0, 3, 3, 0], ['self', 2, 1, 3, 1], ['setLog',0,0]], ['code603', 'code604', 'code604', 'code604'])
# insList = [['trace', 0, 0, 1, 0], ['self', 0, 1, 1, 1], ['self', 0, 2, 1, 2], ['self', 0, 3, 1, 3], ['self', 0, 4, 1, 4]]         

tnList = ['code603','code603', 'code603', 'code603']
insList = [['trace', 0, 3, 1, 1], ['self', 0, 4, 1, 0], ['trace', 1, 3, 2, 1], ['self', 0, 1, 2, 0], ['trace', 1, 4, 3, 1],  ['setLog', 0, 0]]
prog13_1_4 = (insList, tnList)

insList=[['trace', 0, 3, 1, 0], ['trace', 1, 1, 2, 5], ['trace', 0, 4, 3, 0], ['trace', 0, 2, 4, 0], ['self', 2, 4, 3, 1],['self', 2, 2, 4, 1],  ['setLog', 0, 5]] 
tnList = ['code603', 'codeS', 'code603', 'codeH', 'codeH']  
prog513 = (insList, tnList)

progStean = ([['trace', 0, 0, 1,0], ['self', 0, 1, 1, 1], ['setLog', 0, 2]], ['code603', 'code603'])

# 713 4 4 code
prog713_4_4 = ([['trace', 0, 5, 1, 0], ['trace', 0, 0, 2, 0], ['trace', 0, 1, 3, 0], ['trace', 1, 1, 4, 0], ['self', 2, 1, 3, 1], ['setLog', 0, 2]], ['code603', 'code604', 'codeH', 'codeH', 'codeS'])
# n: 7, k: 1, d: 3, Ks: 1, rW: 4, cW: 4, error: 1.08502e-4
#713 5 6 code

prog713_5_5 = ([['trace', 0, 0, 1, 0], ['trace', 0, 2, 2, 0], ['trace', 0, 5, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 1, 1, 1], ['setLog', 0, 4]], ['code603', 'code604', 'codeH', 'codeH', 'codeS'])
# n: 7, k: 1, d: 3, Ks: 1, rW: 5, cW: 5, error: 3.33403e-5

prog713_6_6 = ([['trace', 0, 0, 1, 0], ['trace', 0, 5, 2, 0], ['trace', 1, 1, 3, 0], ['trace', 3, 1, 4, 0], ['self', 0, 1, 1, 2], ['setLog', 0, 2]], ['code603', 'code604', 'codeH', 'codeH', 'codeS'])

prog813_44 = ([['trace', 0, 0, 1, 0], ['trace', 1, 2, 2, 0], ['trace', 1, 5, 3, 0], ['trace', 0, 1, 4, 0], ['trace', 1, 3, 5, 0], ['self', 2, 1, 3, 1], ['setLog', 0, 2]], ['code604', 'code603', 'codeH', 'codeH', 'codeS', 'codeGHZ'])
prog813_57 = ([['trace', 0, 0, 1, 0], ['trace', 1, 2, 2, 0], ['trace', 0, 1, 3, 0], ['trace', 1, 1, 4, 0], ['trace', 0, 2, 5, 0], ['self', 0, 3, 4, 1], ['setLog', 0, 4]],['code604', 'code603', 'codeH', 'codeS', 'codeS', 'codeGHZ'])
prog813_66 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 2, 1, 4, 0], ['self', 2, 2, 4, 1], ['setLog', 0, 3]],['code604', 'codeS', 'code604', 'codeS', 'codeGHZ'])

prog422 = ([['setLog', 0, 0], ['setLog', 0, 1]],['code603'])
prog422_trick = ([['trace', 0, 2, 1, 2], ['self', 0, 3, 1, 3], ['setLog', 1, 0], ['setLog', 0, 0],['setLog', 0, 1],['setLog', 1, 1]], ['code603', 'code603'])

prog6_1_3_54=([['trace', 0, 0, 1, 0], ['trace', 1, 1, 2, 0], ['trace', 1, 2, 3, 0], ['self', 1, 3, 2, 1], ['self', 1, 5, 3, 1], ['setLog', 0, 1]], ['code604', 'code603', 'codeH', 'codeGHZ'])

prog6_1_3_55 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 2, 1, 3, 0], ['trace', 2, 2, 4, 0], ['trace', 0, 2, 5, 0], ['setLog', 0, 3]], ['code604', 'codeS', 'codeGHZ', 'codeH', 'codeH', 'codeS'])

tmp = [['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 4, 4, 1], ['self', 0, 5, 3, 1], ['setLog', 1, 1], ['setLog', 2, 1]]
#  ['code604', 'code604', 'code604', 'codeS', 'codeH']
prog823_43 = (tmp ,['code604', 'code604', 'code604', 'codeS', 'codeH'])
tnList = ['code604',  'codeS']
insList = [['trace', 0, 0, 1, 0],   ['self', 0, 1, 1, 1]]
debug = (insList, tnList)

prog823_54 = ([['trace', 0, 0, 1, 0], ['trace', 1, 1, 2, 0], ['trace', 1, 2, 3, 0], ['trace', 3, 1, 4, 0], ['self', 1, 3, 2, 1], ['self', 2, 2, 4, 1], ['setLog', 0, 1], ['setLog', 2, 3]], ['code604', 'code604', 'code604', 'codeH', 'codeS'])
prog823_43_2 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0],['self', 0, 5, 2, 1], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 4, 3, 1],  ['setLog', 1, 1], ['setLog', 4, 1]],['code604', 'code604', 'codeH', 'codeH', 'code604'])
prog923_44 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 4, 4, 1], ['self', 0, 5, 3, 1], ['setLog', 1, 1], ['setLog', 2, 1]],['code604', 'code604', 'code604', 'codeGHZ', 'codeH'])

prog513_44 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['setLog', 0, 2]],['code604', 'codeS', 'codeS'])
prog513_43= ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['self', 0, 3, 3, 1], ['self', 0, 4, 2, 1], ['setLog', 0, 5]], ['code603', 'code604', 'codeH', 'codeS'])

prog16_2_3 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 2],['self',0,4,2,1],['self',0,5,1,1], ['setLog', 2, 3], ['setLog', 3, 1]],['code603', 'codeH', 'code604', 'code604', 'code603'])

prog623_trick = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['self', 0, 2, 2, 1], ['setLog', 0, 3], ['setLog', 1, 1]], ['code603', 'code604', 'codeS'])

prog913_4_4 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 3, 1, 4, 0], ['self', 1, 1, 2, 1], ['setLog', 0, 3]], ['code804', 'codeH', 'codeH', 'code604', 'codeS'])
# n: 9, k: 1, d: 3, Ks: 1, rW: 4, cW: 4, error: 9.62497e-5

prog913_6_6 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 2, 1, 3, 0], ['self', 0, 2, 1, 1], ['setLog', 0, 3]], ['code804', 'codeH', 'code604', 'codeH'])

prog913_8_7 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 3, 1, 4, 0], ['self', 1, 1, 2, 1], ['setLog', 0, 3]], ['code804', 'codeH', 'codeS', 'code604', 'codeS'])

prog1013_6_7 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 2, 1, 3, 0], ['trace', 3, 1, 4, 0], ['self', 0, 2, 1, 1], ['self', 3, 2, 4, 1], ['setLog', 0, 3]], ['code604', 'code603', 'codeH', 'codeGHZ', 'code604'])

prog12_3_106 = ([['trace', 0, 0, 1, 0], ['trace', 1, 2, 2, 0], ['trace', 1, 4, 3, 0], ['trace', 0, 1, 4, 0], ['trace', 0, 2, 5, 0], ['self', 0, 3, 1, 3], ['setLog', 0, 4]], ['code604', 'code603', 'codeH', 'codeS', 'codeGHZ', 'code604'])

prog13_1_4_78 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 4, 4, 1], ['self', 2, 1, 3, 1], ['setLog', 0, 5]],['code604', 'code603', 'codeH', 'code604', 'code604'])

prog13_1_4_99 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 2, 1, 3, 0], ['self', 0, 2, 3, 1], ['self', 1, 1, 2, 2], ['setLog', 0, 3]], ['code604', 'code604', 'code604', 'code603'])

prog11_1_3_98 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 4, 1, 1], ['setLog', 0, 5]],['code604', 'code603', 'codeH', 'codeH', 'code604'])

prog12_3_108 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 1, 2, 4, 0], ['trace', 0, 3, 5, 0], ['self', 0, 4, 1, 1], ['setLog', 0, 5]], ['code604', 'code603', 'codeH', 'codeH', 'codeGHZ', 'code604'])

prog13_1_4_1010 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 1, 1, 3, 0], ['trace', 0, 2, 4, 0], ['self', 0, 3, 3, 1], ['self', 3, 2, 4, 1], ['setLog', 0, 4]], ['code604', 'code603', 'codeH', 'code604', 'code604'])

prog7_3_66 = ([['trace', 0, 0, 1, 0], ['trace', 1, 2, 2, 0], ['trace', 0, 1, 3, 0], ['trace', 0, 2, 4, 0], ['trace', 0, 3, 5, 0], ['self', 1, 5, 2, 1], ['setLog', 0, 4]], ['code604', 'code603', 'codeH', 'codeS', 'codeS', 'codeH'])

tnList = ['code603', 'code0', 'codePlus'] + ['code603', 'code0'] + ['code603', 'code0', 'codePlus'] + ['code603', 'codePlus'] + ['code603'] + ['code603', 'codePlus'] + ['code603', 'code0', 'codePlus'] + ['code603', 'code0'] + ['code603', 'code0', 'codePlus']
tens6 =  [0, 3, 5, 8, 10, 11, 13, 16, 18]
instList = [['trace', 0, 5, 1, 0], ['trace', 0, 2, 2, 0], ['trace', 0, 4, 3, 4]] 
instList += [['trace', 3, 5, 4, 0]]#, ['trace', 3, 2, 5, 2]] 
# instList += [['trace', 5, 5, 6, 0], ['trace', 5, 4, 7, 0], ['trace', 5, 3, 8, 5]]
# instList += [['trace', 8, 2, 9, 0], ['trace', 8, 4, 10, 4]]
# instList += [['self', 10, 5, 3, 3], ['trace', 10, 2, 11, 2]]
# instList += [['self', 11, 5, 0, 3], ['trace', 11, 4, 12, 0], ['trace', 11, 3, 13, 5]]
# instList += [['trace', 13, 3, 14, 0], ['trace', 13, 2, 15, 0], ['trace', 13, 4, 16, 4]]
# instList += [['self', 16, 5, 10, 3], ['trace', 16, 3, 17, 0], ['trace', 16, 2, 18, 2]]
# instList += [['self', 18, 5, 8, 3], ['trace', 18, 3, 19, 0], ['trace', 18, 4, 20, 0]]

# instList = [['trace', 0, 2, 1, 3],['trace', 1, 5, 2, 4],['trace', 0, 3, 3, 4],['trace', 3, 5, 4, 4],['trace', 4, 2, 5, 3],['trace', 3, 2, 6, 5],['trace', 6, 2, 7, 3],['trace', 7, 5, 8, 4],['trace', 0, 4, 9, 0],['trace', 3, 3, 10, 0],['trace', 6, 4, 11, 0],['trace', 2, 2, 12, 0],['trace', 5, 5, 13, 0], ['trace', 8, 2, 14, 0],['trace', 0, 5, 15, 0],['trace', 1, 4, 16, 0],['trace', 2, 5, 17, 0],['trace', 6, 3, 18, 0],['trace', 7, 2, 19, 0],['trace', 8, 3, 20, 0]]
# instList += [['self', 1, 2, 4, 5],['self', 2, 3, 5, 4],['self', 4, 3, 7, 4],['self', 5, 2, 8, 5]]
tens6u = [0,  5,  10,  13,  18]
tens6d = [ 3, 8, 11, 16]
instList += [['setLog', i, 1] for i in tens6[:2]]
prog3t3_surface = (instList, tnList) 

tnList = ['code603', 'code0', 'codePlus']  + ['code603', 'code0', 'codePlus'] + ['code603', 'code0', 'codePlus'] + ['code603', 'code0', 'codePlus']
tens6 =  [0, 3, 6, 9]
instList = [['trace', 0, 5, 1, 0], ['trace', 0, 2, 2, 0], ['trace', 0, 4, 3, 4]] 
instList += [['trace', 3, 5, 4, 0], ['trace', 3, 2, 5, 0], ['trace', 3, 3, 6, 5]] 
instList += [['trace', 6, 3, 7, 0], ['trace', 6, 4, 8, 0], ['trace', 6, 2, 9, 2]] 
instList += [['self', 9, 5, 0, 3], ['trace', 9, 3, 10, 0], ['trace', 9, 4, 11, 0]] 
instList += [['setLog', i, 1] for i in tens6]

prog2t2_surface = (instList, tnList) 

instList = [['setLog', 0, 0]]
tnList = ['code603']
prog512 = (instList, tnList)

tnList = ['code603', 'code0', 'codePlus']
instList = [['trace', 0, 5, 1, 0], ['trace', 0, 2, 2, 0], ['setLog', 0, 0]] 
debug = (instList, tnList)

if __name__ == "__main__":

    prog = prog513_44

    candidates = [prog513_44, prog6_1_3_54, prog7_3_66, prog813_66, prog913_6_6, prog1013_6_7, prog11_1_3_98,prog12_3_108, prog13_1_4_1010]
    candidateProg = [prog513_44, prog6_1_3_54, prog7_3_66, prog813_66, prog913_6_6, prog1013_6_7, prog11_1_3_98,prog12_3_108, prog13_1_4_1010]
    for prog in candidates:
        px = 0.01
        pz = 0.05
        eval_prog(prog, px, pz)
        # stabs = [stab.toInt() for stab in prog2Stab(prog)]
        # print(stabs)
        print("")


# a = prog2Cm(debug[0], [eval(t) for t in debug[1]])
# a.row_echelon()
# b = prog2Cm(prog[0], [eval(t) for t in prog[1]])
# b.row_echelon()
# print(b.matrix)
# index = [4,5,6,7,0,1,2,3]
# tmp = [i + 8 for i in index]
# index += tmp
# c = b.matrix[:, index]

# b.matrix = c
# b.row_echelon()
# print(b.matrix)
# print(a.matrix)

# eval_prog(prog823_43)
# print(simp_poly(get_enum_tensor(code823, [])[0]))


# cm = check_matrix(code603)
# a = cm.trace(check_matrix(code604),0,0).trace(check_matrix(code604),1,0).selfTrace(3, 9)
# a = check_matrix(code603).trace(check_matrix(codeS),4,0).trace(check_matrix(codeH),3,0).trace(check_matrix(codeH),3,0).trace(check_matrix(code603),3,4).selfTrace(3, 8).selfTrace(3, 7)
# a.setLogical(2)

# a.setLogical(2)
# a.setLogical(4)

# print(f"n: {a.n}, d: {distance(code, 1)}")
# import time
# start = time.time()
# stab_group = stabilizer_group(code)
# A = Azx(stab_group, px, 1 - px, pz, 1- pz)
# B = Bzx(1, stab_group, px, 1 - px, pz, 1- pz)
# print(A, B, B-A)
# end = time.time()
# print(f"time: {end-start}")



# code = code11_1_5
# stab_group = stabilizer_group(code)

# d=distance(code, 2, stab_group)
# print(f"distance: {d}, n: {code[0].length}")

# A = Azx(stab_group, px, 1 - px, pz, 1- pz)
# B = Bzx(2, stab_group, px, 1 - px, pz, 1- pz)
# print(A, B, B-A)
# exit(0)
 

# code = codelize(['xxxx', 'zzzz'])
# cm = check_matrix(code603, symmetry=[[0,1,2,3], [4,5]])
# print(cm.matrix)
# print(cm.symmetry)
# a = cm.trace(cm, 0, 4)
# b = a.trace(cm, 0, 4)
# c = b.trace(cm, 0, 4)
# d = c.trace(cm, 0, 4)
# d.setLogical(1)
# # print(d.matrix)
# print(d.matrix.shape)
# code = checkM2Stabilizers(d.matrix)
# print(code)
# print(distance(code, 1))
# exit(0)2

# a = cm.trace(cm, 0,0)
# a.selfTrace(0, 7)
# a.setLogical(2)
# # a.setLogical(2)
# # a.setLogical(4)
# a.setLogical(4)
# print(a.matrix)
# code = checkM2Stabilizers(a.matrix)
# print(code)
# print(distance(code, 2))
# exit(0)

# a = cm.trace(cm, 4,4)
# print(a.symmetry)

# steane = a.selfTrace(4,9)
# print(steane.symmetry)
# print(steane.matrix)
# LogicOp = steane.setLogical(0)
# print(steane.matrix)
# steane = checkM2Stabilizers(steane.matrix)
# print(steane)
# print(distance(steane, 1))
# exit(0)

# b = a.trace(cm, 9, 0)
# print("b")
# c = b.trace(cm, 11, 4)
# print("c")
# d = c.selfTrace(1, 12)
# print("d")
# e = d.selfTrace(0, 10)
# e = cm.trace(cm,0,0).trace(cm, 2, 0)
# value_range=[
#     list(range(14)),
#     [0,4],
#     list(range(18)),
#     list(range(18)),
#     list(range(16)),
#     list(range(16)),
#     [2,3,4]
# ]
# for v in itertools.product(*value_range):
#     if v[2]>=v[3] or v[4]>=v[5]:
#         continue
#     print(v)
#     c = b.trace(cm, v[0], v[1])
#     d = c.selfTrace(v[2], v[3]).selfTrace(v[4],v[5])
#     d.setLogical(v[6])
#     code = checkM2Stabilizers(d.matrix)
#     dis=distance(code, 1)
#     print(f"distance: {dis}")
#     if dis >= 4:
#         exit(0)


# e.setLogical(2)
# code = checkM2Stabilizers(e.matrix)
# stb_group = stabilizer_group(code)
# print(e.matrix)




# e = cm.trace(cm,0,0).trace(cm, 2, 0)
# e.setLogical(2)
# a = cm.trace(cm, 0, 4).selfTrace(0, 9)
# b = a.trace(cm, 4, 5).trace(cm, 4, 5).trace(cm, 4,4).trace(cm,1,4)
# c = b.selfTrace(6,13).selfTrace(6,16).selfTrace(1,18)
# c.setLogical(2)
# print("c", c.matrix.shape)
# print(c.matrix)
# code = checkM2Stabilizers(c.matrix)

# code17_1_3 = codelize(["xiiiiiixxiixxixix","ixiiiiiiiixixixix","iixixiiiiiixxiiii","iiixxiiiiiixxixxi","iiiiixixiixxiiixx","iiiiiixixixxiiixx","iiiiiiiiixxxxiiii","iiiiiiiiiiiiixxxx","ziiiiizziiiiiiiii","iziiiiizziizziiii","iiziiiiiiiziziizz","iiiziiizziiiiizzi","iiiiziizziziziziz","iiiiizzzziiiiiiii","iiiiiiiiizzzziiii","iiiiiiiiiiiiizzzz"])


# print(distance(code13_1_4, 1))
# print(check_matrix(code13_1_4).matrix)

