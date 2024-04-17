import numpy as np
from interactive import *
import pickle

def circ_dis(M):
    pass
def writeMatrix(path, M):
    save_dir = './'
    shape = M.shape
    head = "%%MatrixMarket matrix coordinate integer general\n%% Field: GF(2)\n\n"
    with open(save_dir+path, 'w') as f:
        f.write(head)
        f.write(f"{shape[0]} {shape[1]} {2*shape[1]}\n")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if M[i, j]==1:
                    f.write(f"{i+1} {j+1} 1\n")

def Get_circ_d(proc, Hx, Hz, l=0, m=0):

    writeMatrix(f'{l}{m}Hx.mtx', Hx)
    writeMatrix(f'{l}{m}Hz.mtx', Hz)
    write(proc, f'lisX:=ReadMTXE("{l}{m}Hx.mtx",0);;')
    write(proc, f'lisZ:=ReadMTXE("{l}{m}Hz.mtx",0);;')
    write(proc, 'Read("EvalDis.g");')
    write(proc, 'd;')
    d = int(read(proc))
    return d


error_rate = 0.003
# code parameters and number of syndrome cycles
n = 144
k = 12
d = 12
num_cycles = 12

title = './TMP/mydata_' + str(n) + '_' + str(k) + '_p_' + str(error_rate) + '_cycles_' + str(num_cycles)
print('reading data from file')
print(title)
with open(title, 'rb') as fp:
	mydata = pickle.load(fp)

HX = mydata['HdecX'].toarray()
HZ = mydata['HdecZ'].toarray()


print(HX.shape)