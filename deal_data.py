from enumerator import *
import os
print(os.getcwd())
dir_path = os.getcwd()+"\\zaratan\\08.22\\"

res = []

# Iterate directory
result = ""
for file_path in os.listdir(dir_path):
    # check if current file_path is a file

    if os.path.isfile(os.path.join(dir_path, file_path)):
    # add filename to list
        print(file_path)     
        with open(dir_path+file_path, 'r') as f:
            content=f.readlines()[:-1]
            count = len(content)//4
            for i in range(count):
                if "KS:2" in content[4*i+2]:
                    ins = eval(content[4*i].strip())
                    code = eval(content[4*i+1].strip())
                    prog = (ins, code)
                    insList = prog[0]
                    tnList = prog[1]
                    tensorList  = [eval(t) for t in tnList]
                    a = prog2Cm(insList, tensorList)
                    tn = prog2TNN(insList, tnList)
                    n = tn.get_n()
                    k = tn.get_k()

                    tmp = tn.toCm()
                    tmp.row_echelon()
                    rw = tmp.rowWBound()
                    cw = tmp.colWBound()
                    # tn.setLogical(0,0)

                    d,error,K = eval_TN(tn)
                    lines = content[4*i]+content[4*i+1]+f"n: {n}, d: {d}, K:{K}, rW: {rw}, cW: {cw} error: {error}\n\n"
                    result+=lines

with open("output", "w") as out:
    out.write(result)

