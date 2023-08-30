from adt import *
import itertools
from enumerator import *

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

px = 0.01
pz = 0.05

prog15_14 = ([['trace', 0, 2, 1, 0], ['trace', 0, 4, 2, 0], ['trace', 0, 3, 3, 0], ['self', 2, 1, 3, 1], ['setLog',0,0]], ['code603', 'code604', 'code604', 'code604'])
# insList = [['trace', 0, 0, 1, 0], ['self', 0, 1, 1, 1], ['self', 0, 2, 1, 2], ['self', 0, 3, 1, 3], ['self', 0, 4, 1, 4]]         
tnList = ['code604', 'code604']
tnList = ['code603','codeS', 'codeH', 'codeH', 'code603']
insList = [['trace', 0, 4, 1, 0], ['trace', 0, 3, 2, 0], ['trace', 0, 5, 3, 0],['trace', 1, 1, 4, 4], ['self', 4, 3, 2, 1], ['self', 4, 5, 3, 1], ['setLog', 0, 2]]
insList=[['trace', 0, 0, 1, 0], ['trace', 0, 4, 2, 0], ['trace', 1, 1, 3, 0], ['trace', 0, 1, 4, 0], ['self', 3, 1, 4, 1], ['setLog', 0, 2]] 
tnList = ['code603', 'code603', 'codeH', 'codeH', 'codeS']  

progStean = ([['trace', 0, 0, 1,0], ['self', 0, 1, 1, 1], ['setLog', 0, 2]], ['code603', 'code603'])

# 713 4 6 code
prog713_4_4 = ([['trace', 0, 5, 1, 0], ['trace', 0, 0, 2, 0], ['trace', 0, 1, 3, 0], ['trace', 1, 1, 4, 0], ['self', 2, 1, 3, 1], ['setLog', 0, 2]], ['code603', 'code604', 'codeH', 'codeH', 'codeS'])

#713 5 6 code

prog713_5_5 = ([['trace', 0, 0, 1, 0], ['trace', 0, 2, 2, 0], ['trace', 0, 5, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 1, 1, 1], ['setLog', 0, 4]], ['code603', 'code604', 'codeH', 'codeH', 'codeS'])

prog713_6_6 = ([['trace', 0, 0, 1, 0], ['trace', 0, 5, 2, 0], ['trace', 1, 1, 3, 0], ['trace', 3, 1, 4, 0], ['self', 0, 1, 1, 2], ['setLog', 0, 2]], ['code603', 'code604', 'codeH', 'codeH', 'codeS'])

prog422 = ([['setLog', 0, 0], ['setLog', 0, 1]],['code603'])
prog422_trick = ([['trace', 0, 2, 1, 2], ['self', 0, 3, 1, 3], ['setLog', 1, 0], ['setLog', 0, 0],['setLog', 0, 1],['setLog', 1, 1]], ['code603', 'code603'])


tmp = [['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 4, 4, 1], ['self', 0, 5, 3, 1], ['setLog', 1, 1], ['setLog', 2, 1]]
#  ['code604', 'code604', 'code604', 'codeS', 'codeH']
prog823_43 = (tmp ,['code604', 'code604', 'code604', 'codeS', 'codeH'])
tnList = ['code604',  'codeS']
insList = [['trace', 0, 0, 1, 0],   ['self', 0, 1, 1, 1]]
debug = (insList, tnList)

prog823_54 = ([['trace', 0, 0, 1, 0], ['trace', 1, 1, 2, 0], ['trace', 1, 2, 3, 0], ['trace', 3, 1, 4, 0], ['self', 1, 3, 2, 1], ['self', 2, 2, 4, 1], ['setLog', 0, 1], ['setLog', 2, 3]], ['code604', 'code604', 'code604', 'codeH', 'codeS'])
prog823_43_2 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 4, 3, 1], ['self', 0, 5, 2, 1], ['setLog', 1, 1], ['setLog', 4, 1]],['code604', 'code604', 'codeH', 'codeH', 'code604'])
prog923_44 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 0], ['self', 0, 4, 4, 1], ['self', 0, 5, 3, 1], ['setLog', 1, 1], ['setLog', 2, 1]],['code604', 'code604', 'code604', 'codeGHZ', 'codeH'])

prog16_2_3 = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['trace', 0, 2, 3, 0], ['trace', 0, 3, 4, 2],['self',0,4,2,1],['self',0,5,1,1], ['setLog', 2, 3], ['setLog', 3, 1]],['code603', 'codeH', 'code604', 'code604', 'code603'])

prog623_trick = ([['trace', 0, 0, 1, 0], ['trace', 0, 1, 2, 0], ['self', 0, 2, 2, 1], ['setLog', 0, 3], ['setLog', 1, 1]], ['code603', 'code604', 'codeS'])

prog = prog823_54

def eval_prog(prog):


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

    d,error,K = eval_TN(tn)

    print(f"n: {n}, k: {k}, d: {d}, Ks: {K}, rW: {rw}, cW: {cw}, error: {error:.4e}")

    code = checkM2Stabilizers(tmp.matrix)
    print(code)
    print(tmp.matrix)
    # for op in tmp.LogicOp:
    #     print(op.shape[1]//2)
    #     print(op)

    stab_group = stabilizer_group(code)

    print("ABzx", ABzx(stab_group, px, 1 - px, pz, 1- pz, k, K))
    # print(eval_tn(tn))
    print(distance(code, k, stab_group))

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
eval_prog(prog)
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
exit(0)


code = code11_1_5
stab_group = stabilizer_group(code)

d=distance(code, 2, stab_group)
print(f"distance: {d}, n: {code[0].length}")

A = Azx(stab_group, px, 1 - px, pz, 1- pz)
B = Bzx(2, stab_group, px, 1 - px, pz, 1- pz)
print(A, B, B-A)
exit(0)
 

code = codelize(['xxxx', 'zzzz'])
cm = check_matrix(code603, symmetry=[[0,1,2,3], [4,5]])
print(cm.matrix)
print(cm.symmetry)
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

a = cm.trace(cm, 4,4)
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

b = a.trace(cm, 9, 0)
print("b")
c = b.trace(cm, 11, 4)
print("c")
d = c.selfTrace(1, 12)
print("d")
e = d.selfTrace(0, 10)
e = cm.trace(cm,0,0).trace(cm, 2, 0)
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


e.setLogical(2)
code = checkM2Stabilizers(e.matrix)
# stb_group = stabilizer_group(code)
print(e.matrix)




e = cm.trace(cm,0,0).trace(cm, 2, 0)
e.setLogical(2)
a = cm.trace(cm, 0, 4).selfTrace(0, 9)
b = a.trace(cm, 4, 5).trace(cm, 4, 5).trace(cm, 4,4).trace(cm,1,4)
c = b.selfTrace(6,13).selfTrace(6,16).selfTrace(1,18)
c.setLogical(2)
print("c", c.matrix.shape)
print(c.matrix)
code = checkM2Stabilizers(c.matrix)

code17_1_3 = codelize(["xiiiiiixxiixxixix","ixiiiiiiiixixixix","iixixiiiiiixxiiii","iiixxiiiiiixxixxi","iiiiixixiixxiiixx","iiiiiixixixxiiixx","iiiiiiiiixxxxiiii","iiiiiiiiiiiiixxxx","ziiiiizziiiiiiiii","iziiiiizziizziiii","iiziiiiiiiziziizz","iiiziiizziiiiizzi","iiiiziizziziziziz","iiiiizzzziiiiiiii","iiiiiiiiizzzziiii","iiiiiiiiiiiiizzzz"])


# print(distance(code13_1_4, 1))
# print(check_matrix(code13_1_4).matrix)

