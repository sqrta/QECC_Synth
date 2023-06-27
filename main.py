from adt import *
import itertools
from enumerator import *

px = 0.01
pz = 0.05
cm = check_matrix(code603)
cm604 = check_matrix(code604)
# a = cm.trace(check_matrix(code604),0,0).trace(check_matrix(code604),1,0).selfTrace(3, 9)
a = cm.trace(check_matrix(code604),0,0).trace(check_matrix(code604),0,0).selfTrace(3, 4)
a.setLogical(0)
# a.setLogical(2)
# a.setLogical(4)
print(a.matrix)
code = checkM2Stabilizers(a.matrix)
print(code)
print(distance(code, 1))
stab_group = stabilizer_group(code)
A = Azx(stab_group, px, 1 - px, pz, 1- pz)
B = Bzx(1, stab_group, px, 1 - px, pz, 1- pz)
print(A, B, B-A)
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

