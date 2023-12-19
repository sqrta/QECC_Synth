from enumerator import *
from adt import *
from progEval import *
import os

def getCode(filename="tmp"):
    k = 1
    with open(filename, "r") as f:
        lines = f.readlines()
        xlist = []
        zlist = []
        for line in lines:
            xz = line.split('|')
            x = xz[0].replace(' ','').replace("1", 'x').replace('0', 'i').replace('[', '')
            z = xz[1].replace(' ','').replace("1", 'z').replace('0', 'i').replace(']', '').replace('\n', '')
            xlist.append(x)
            zlist.append(z)
        px = 0.01
        pz = 0.05
        code = codeTN(xlist+zlist)
        code.merge()
    return code

def aberror(code, k):
    stab_group = stabilizer_group(code)
    d = distance(code, k, stab_group)
    error = ABzx(stab_group, px, 1 - px, pz, 1- pz, k)
    cm = check_matrix(code)
    print(error)
    print(f"error: {error[1]:.5e}, n:{code.length}, d: {d}, rowW: {cm.rowWBound()}, colW: {cm.colWBound()}")



def evalCodeFile(file, px, pz, k, K):
    code = getCode(file)
    stabs = [stab.toInt() for stab in code]
    # return stabs
    cm = check_matrix(code)
    xe, ze = ErrorFrCode(px, pz, cm)
    stabs = [stab.toInt() for stab in code]
    # aberror(code, k)
    return stabs, xe, ze
    # 
    # evalFromCeckKMatrix(cm, px, pz, k, K)

if __name__ == "__main__":
    dir_path = "exist_code/"
    files = [file for file in os.listdir(dir_path)]
    vars = ["code_"+file for file in os.listdir(dir_path)]
    px = 0.001
    pz = 0.005
    k = 1
    K = 1
    print(f"[{','.join(vars)}]")
    for file in files:
        path = dir_path + file
        stabs, xe, ze = evalCodeFile(path, px, pz, k, K)
        print(f"code_{file}=CodeWithError({stabs},{xe},{ze})")
