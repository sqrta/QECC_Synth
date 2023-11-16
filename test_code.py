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
    print(f"error: {error}, n:{code.length}, d: {d}, rowW: {cm.rowWBound()}, colW: {cm.colWBound()}")

def evalCodeFile(file, px, pz, k, K):
    code = getCode(file)
    cm = check_matrix(code)
    evalFromCeckKMatrix(cm, px, pz, k, K)

if __name__ == "__main__":
    dir_path = "exist_code/"
    files = [dir_path + file for file in os.listdir(dir_path)]
    px = 0.01
    pz = 0.05
    k = 1
    K = 1
    for file in files:
        print(file)
        evalCodeFile(file, px, pz, k, K)
        print("")
    