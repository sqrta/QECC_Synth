from adt import *

code513 = codelize(["xzzxi", 'ixzzx', 'xixzz', 'zxixz'])


code422 = codelize(['xxxx', 'zzzz'])
code603 = codelize(['xxxxii', 'zzzzii', 'xxiixi', 'ixixix', 'zizizi', 'iizziz'])
code = codelize(['xxxx', 'zzzz'])
cm = check_matrix(code603)
print(cm.matrix)
# a = cm.trace(cm, 0, 4)
# b = a.trace(cm, 0, 4)
# c = b.trace(cm, 0, 4)
# d = c.trace(cm, 0, 4)
# d.setLogical(1)
# print(d.matrix)
# code = checkM2Stabilizers(d.matrix)
# print(code)
# print(distance(code, 1))
# exit(0)

a = cm.trace(cm, 4,4)

# steane = a.selfTrace(4,9)
# print(steane.matrix)
# LogicOp = steane.setLogical(0)
# print(steane.matrix)
# steane = checkM2Stabilizers(steane.matrix)
# print(steane)
# print(distance(steane, 1))
# exit(0)
b = a.trace(cm, 9, 0)
print(f"b:{b.n}")
# for i in range(5,14):
#     for j in range(5,11):
c = b.selfTrace(0, 12)
d = c.selfTrace(0, 11)

e = d.trace(cm, 9, 5)
e.setLogical(1)
code = checkM2Stabilizers(e.matrix)

print(distance(code, 1))
# for i in range(9):
#     for j in range(i,10):
        
#         b=copy.deepcopy(a)
#         steane = a.selfTrace(i,j)
#         for k in range(5):
#             tmp=copy.deepcopy(steane)
#             tmp.setLogical(k)
#             code = checkM2Stabilizers(tmp.matrix)
#             print(distance(code,1))
#             if distance(code, 1)>=3:
#                 print(i,j,k)
#                 print(code)

code13_1_3 = codelize(['xiixixxiiixxi', 'ixixixixiiiii',
'iixxiixxiixxi',
'iiiixxxxiiiii',
'iiiiiiiixiixx',
'iiiiiiiiixxxx','ziiziizziiizz','izizizziiiiii', 'iizziziziiizz','iiiizzzziiiii', 'iiiiiiiiziziz', 'iiiiiiiiizzzz'])
print(distance(code13_1_3, 1))
