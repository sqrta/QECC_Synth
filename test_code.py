from enumerator import *
from adt import *

filename = "tmp"
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
    code = codeTN(xlist+zlist)
    code.merge()
    d,error = eval_code(code, 1)
    print(f"d: {d}, error: {error}")