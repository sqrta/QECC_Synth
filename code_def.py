from adt import *

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
    def __init__(self, tensorName, index = 0) -> None:
        self.size = eval(tensorName).length
        self.tracted = []
        self.name = tensorName
        self.index = index

    def tensor(self):
        return eval(self.name)

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
            print(index1, leg1, index2, leg2)
            raise ValueError(leg1, leg2, tn1.tracted, tn2.tracted)
        tn1.tracted.append(leg1)
        tn2.tracted.append(leg2)
        self.insList.append(["self", index1, leg1, index2, leg2])
        self.selfTraceCount+=1

    def setLogical(self, index, leg):
        self.tensorList[index].tracted.append(leg)
        self.Logical.append((index,leg))
        self.insList.append(["setLog", index, leg])

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
    
    def toCm(self):
        tnList = [t.tensor() for t in self.tensorList]
        insList = self.insList
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
            elif ins[0] == "self":
                index1, leg1, index2, leg2 = ins[1:]
                mIndex1 = getMIndex(index1, leg1)
                mIndex2 = getMIndex(index2, leg2)

                cm = cm.selfTrace(mIndex1, mIndex2)
                tracted[index1].append(leg1)
                tracted[index2].append(leg2)
            elif ins[0] == "setLog":
                index1, leg1 = ins[1:]
                mIndex1 = getMIndex(index1, leg1)
                cm.setLogical(mIndex1)
                tracted[index1].append(leg1)
            else:
                raise NameError(f"no ops as {ins[0]}")
        cm.removeZeroRow()
        return cm

code13_1_4 = codeTN(['xiixixxiiixxi', 'ixixixixiiiii',
'iixxiixxiixxi',
'iiiixxxxiiiii',
'iiiiiiiixiixx',
'iiiiiiiiixxxx','ziiziizziiizz','izizizziiiiii', 'iizziziziiizz','iiiizzzziiiii', 'iiiiiiiiziziz', 'iiiiiiiiizzzz'], "code13_1_4")
code17_1_3 = codeTN(["xiiiiiixxiixxixix","ixiiiiiiiixixixix","iixixiiiiiixxiiii","iiixxiiiiiixxixxi","iiiiixixiixxiiixx","iiiiiixixixxiiixx","iiiiiiiiixxxxiiii","iiiiiiiiiiiiixxxx","ziiiiizziiiiiiiii","iziiiiizziizziiii","iiziiiiiiiziziizz","iiiziiizziiiiizzi","iiiiziizziziziziz","iiiiizzzziiiiiiii","iiiiiiiiizzzziiii","iiiiiiiiiiiiizzzz"])
code513 = codeTN(["xzzxi", 'ixzzx', 'xixzz', 'zxixz'], "code513")
code513v = codeTN(['XZZIY', 'IYZZX', 'ZIXZY', 'ZZIXX'])
code604 = codeTN(['ixzzxi', 'iixzzx', 'ixixzz', 'izxixz', 'xxxxxx', 'zzzzzz'], "code604", symmetry=[0])
code603 = codeTN(['iixxxx', 'iizzzz', 'xixxii', 'ixixix', 'zizizi', 'iziizz'], "code603", symmetry=[0,2])
code804 = codeTN(['iiiixxxx', 'iixxiixx', 'ixixixix', 'xxxxxxxx','iiiizzzz', 'iizziizz', 'iziziziz', 'zzzzzzzz'], "code803", symmetry=[0, 1])
code713 = codeTN(['iiixxxx', 'ixxiixx', 'xixixix', 'iiizzzz', 'izziizz', 'ziziziz'], "code713")
code422 = codeTN(['xxxx', 'zzzz'], "code422")
code11_1_5 = codeTN(["ziiiiziiiii","iziiiziiiii","iiziiziiiii","iiiziziiiii","iiiizziiiii","iiiiiiziiiz","iiiiiiiziiz","iiiiiiiiziz","iiiiiiiiizz","iiiiiiiiiii","ixixiiixxii","ixixxxiixxi","ixiixiiiixx","ixxxixxiixi","iixxiixxixx","iixxiixxxix","ixixxxxxixx","iiixixixixi","ixixiixixii","xxxxxxiiiii"])
code5_1_3_m = codeTN(["xiixx","ixiix","iixxi", "iiiii","ziziz","ziizz","ziizz","izzzz"])
codeH = codeTN(["xz", "zx", "yy"], "H", symmetry=[0])
codeS = codeTN(["xy", "yx", ], "S", symmetry=[0])
codet = codeTN(['ixiiz', 'izxix', 'ziixi', 'yiizy', 'iiziz'])
codeGHZ = codeTN(['izz','xxx', 'zzi'], "GHZ", symmetry=[0])
code823 = codeTN(['XZZXIIII', 'ZXZIIIII', 'IZXZIIII', 'IIIIXZZX', 'IIIIZXZI', 'IIIIIZXZ'])
code0 = codeTN(['z'])
codePlus = codeTN(['x'])
code922 = codeTN(['XXXIIIIII', 'IIIXXXIII', 'IIIIIIXXX', 'ZIZZIZZIZ', 'IZZZIZZIZ', 'IIIIZZZIZ', 'IIIIIIIZZ'])
code913 = codeTN(['ZZIIIIIII', 'XXIXXIIII', 'IZZIZZIII', 'IIXIIXIII', 'IIIXIIXII', 'IIIZZIZZI', 'IIIIXXIXX', 'IIIIIIIZZ'])
code613=codeTN(['YIZZIX', 'ZXZIZY', 'IZXZZY', 'ZZIYIX', 'IIIIYX'])
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
    code = code713
    print(Az_poly(code))
    print('d', distance(code, 1))
    stab_group = stabilizer_group(code)
    
    print(ABzx(stab_group, px, 1 - px, pz, 1- pz, 1))
    print([stab.toInt() for stab in code.stabs])
    
