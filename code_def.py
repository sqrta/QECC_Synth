from adt import *

code13_1_4 = codeTN(['xiixixxiiixxi', 'ixixixixiiiii',
'iixxiixxiixxi',
'iiiixxxxiiiii',
'iiiiiiiixiixx',
'iiiiiiiiixxxx','ziiziizziiizz','izizizziiiiii', 'iizziziziiizz','iiiizzzziiiii', 'iiiiiiiiziziz', 'iiiiiiiiizzzz'], "code13_1_4")
code17_1_3 = codeTN(["xiiiiiixxiixxixix","ixiiiiiiiixixixix","iixixiiiiiixxiiii","iiixxiiiiiixxixxi","iiiiixixiixxiiixx","iiiiiixixixxiiixx","iiiiiiiiixxxxiiii","iiiiiiiiiiiiixxxx","ziiiiizziiiiiiiii","iziiiiizziizziiii","iiziiiiiiiziziizz","iiiziiizziiiiizzi","iiiiziizziziziziz","iiiiizzzziiiiiiii","iiiiiiiiizzzziiii","iiiiiiiiiiiiizzzz"])
code513 = codeTN(["xzzxi", 'ixzzx', 'xixzz', 'zxixz'], "code513")
code604 = codeTN(["ixzzxi", 'iixzzx', 'ixixzz', 'izxixz', 'xxxxxx', 'zzzzzz'], "code604", symmetry=[0])
code603 = codeTN(['iixxxx', 'iizzzz', 'xixxii', 'ixixix', 'zizizi', 'iziizz'], "code603", symmetry=[0,2])
code804 = codeTN(['iiiixxxx', 'iixxiixx', 'ixixixix', 'xxxxxxxx','iiiizzzz', 'iizziizz', 'iziziziz', 'zzzzzzzz'], "code803")
code713 = codeTN(['iiixxxx', 'ixxiixx', 'xixixix', 'iiizzzz', 'izziizz', 'ziziziz'], "code713")
code422 = codeTN(['xxxx', 'zzzz'], "code422")
code11_1_5 = codeTN(["ziiiiziiiii","iziiiziiiii","iiziiziiiii","iiiziziiiii","iiiizziiiii","iiiiiiziiiz","iiiiiiiziiz","iiiiiiiiziz","iiiiiiiiizz","iiiiiiiiiii","ixixiiixxii","ixixxxiixxi","ixiixiiiixx","ixxxixxiixi","iixxiixxixx","iixxiixxxix","ixixxxxxixx","iiixixixixi","ixixiixixii","xxxxxxiiiii"])
code5_1_3_m = codeTN(["xiixx","ixiix","iixxi", "iiiii","ziziz","ziizz","ziizz","izzzz"])
codeH = codeTN(["xz", "zx", "yy"], "H", symmetry=[0])
codeS = codeTN(["xy", "yx"], "S", symmetry=[0])
codet = codeTN(['ixiiz', 'izxix', 'ziixi', 'yiizy', 'iiziz'])
# stab_list = []
# length = int(len(code11_1_5.stabs) / 2 )
# for i in range(length):
#     stab_list.append(code11_1_5.stabs[i] * code11_1_5[i+length])
# code11_1_5.stabs = stab_list
code5_1_3_m.merge()
code11_1_5.merge()

if __name__ == "__main__":
    px = 0.01
    pz = 0.05
    print(check_matrix(codeH).matrix)
    exit(0)
    print(distance(code11_1_5, 1))
    stab_group = stabilizer_group(code11_1_5)
    print(ABzx(stab_group, px, 1 - px, pz, 1- pz, 1))
    
