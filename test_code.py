from enumerator import *
from adt import *

filename = "tmp"
k = 2
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
    stab_group = stabilizer_group(code)
    d = distance(code, k, stab_group)
    error = ABzx(stab_group, px, 1 - px, pz, 1- pz, k)
    cm = check_matrix(code)
    print(f"error: {error}, d: {d}, rowW: {cm.rowWBound()}, colW: {cm.colWBound()}")