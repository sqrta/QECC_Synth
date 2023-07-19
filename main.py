from adt import *
import itertools
from enumerator import *

def prog2TNN(insList, tensorList):
    progList = []
    tensorList = [eval(t) for t in tensorList]
    for ins in insList:
        if ins[0]=="trace":
            index = ins[3]
            ins[3] = Tensor(tensorList[index])
        progList.append(ins)
    return buildProg(progList, Tensor(tensorList[0]))

def TNN2CM(tn):
    insList = tn.insList
    tnList = tn.tensorList
    cm = check_matrix(tnList[0].tensor)
    def getMIndex(traceIndex):
        return sum([len(t.tracted) for t in tnList[:traceIndex]])
    for ins in insList:
        if ins[0] == "trace":
            traceIndex, traceLeg, newOneIndex, newOneleg = ins[1:]
            matrixIndex = getMIndex(traceIndex)
            newOne = tnList[newOneIndex]
            newTNindex = newOne.tracted.index(newOneleg)    
            # print('trace', matrixIndex, newTNindex)

            tmp = cm.trace(check_matrix(newOne.tensor), matrixIndex, newTNindex)
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
            tracted[traceIndex].append(traceLeg)
            tracted[newOneIndex].append(newOneleg)
        if ins[0] == "self":
            index1, leg1, index2, leg2 = ins[1:]
            mIndex1 = getMIndex(index1, leg1)
            mIndex2 = getMIndex(index2, leg2)
            cm = cm.selfTrace(mIndex1, mIndex2)
            tracted[index1].append(leg1)
            tracted[index2].append(leg2)
    return cm
px = 0.01
pz = 0.05

insList = [['trace', 0, 2, 1, 0], ['trace', 0, 4, 2, 0], ['trace', 0, 3, 3, 0], ['self', 2, 1, 3, 1], ['setLog',0,0]]
tnList = ['code603', 'code604', 'code604', 'code604']
# insList = [['trace', 0, 0, 1, 0], ['self', 0, 1, 1, 1], ['self', 0, 2, 1, 2], ['self', 0, 3, 1, 3], ['self', 0, 4, 1, 4]]         
# tnList = ['code604', 'code604']
tnList = ['code603','codeS', 'codeH', 'codeH', 'code603']
insList = [['trace', 0, 4, 1, 0], ['trace', 0, 3, 2, 0], ['trace', 0, 5, 3, 0],['trace', 1, 1, 4, 2], ['self', 4, 3, 2, 1], ['self', 4, 5, 3, 1], ['setLog', 0, 2]]
tensorList  = [eval(t) for t in tnList]

a = prog2Cm(insList, tensorList)
tn = prog2TNN(insList, tnList)
# tn.setLogical(0,0)
n = tn.get_n()
d,error = eval_tn(tn)
print(f"n: {n}, d: {d}, error: {error}")

# cm = check_matrix(code603)
# a = cm.trace(check_matrix(code604),0,0).trace(check_matrix(code604),1,0).selfTrace(3, 9)
# a = check_matrix(code603).trace(check_matrix(code604),0,0).trace(check_matrix(code604),0,0).selfTrace(3, 4)
a.setLogical(2)

# a.setLogical(2)
# a.setLogical(4)
print(a.matrix)
code = checkM2Stabilizers(a.matrix)
print(code)
print(f"n: {a.n}, d: {distance(code, 1)}")
import time
start = time.time()
stab_group = stabilizer_group(code)
A = Azx(stab_group, px, 1 - px, pz, 1- pz)
B = Bzx(1, stab_group, px, 1 - px, pz, 1- pz)
print(A, B, B-A)
end = time.time()
print(f"time: {end-start}")
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

