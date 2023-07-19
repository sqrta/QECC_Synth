from adt import *
import itertools
from enumerator import *


def eval_prog(prog, initial, px=0.01, pz = 0.05):
    tn = buildProg(prog, initial)
    tn.show()
    return eval_tn(tn)
 

def search(initial):
    import time
    start = time.time()
    queue = [TNNetwork(initial)]
    maxTensor = 4
    selfTraceDepth = 3
    candidate_code = [code603, code604]
    exist_set = set()
    minError = {}
    count = 0
    f = open("found", 'w')
    while len(queue)>0:
        count+=1
        if count%10000==0:
            f.write(str(minError))
            f.close()
            f= open(f"found{count}",'w')
        top = queue.pop(0)
        logLeg = top.equiv_trace_leg()[0]
        setlog = copy.deepcopy(top)
        setlog.setLogical(logLeg[0], logLeg[1])
        try:
            d, error = eval_tn(setlog)
        except:
            print(str(setlog))
            exit(0)
        n = setlog.get_n()
        if d>=3:
            content = str(setlog) + f"error: {error}, n: {n}, d: {d}"
            # print(content)
            if (n,d) not in minError.keys():
                minError[(n,d)] = error
                print(content)
                f.write(content+"\n\n")
            elif error < minError[(n,d)]:
                minError[(n,d)] = error
                print(content)
                f.write(content+"\n\n")
                
        if (d,error) not in exist_set:
            exist_set.add((d,error))
            if len(top.tensorList)<maxTensor and (len(top.insList)<1 or top.insList[-1][0]!="self"):
                dangleLegs = top.equiv_trace_leg()
                for i in range(len(candidate_code)):
                    code = candidate_code[i]
                    if i< top.tensorList[-1].index:
                        continue
                    for leg in dangleLegs:
                        for tractLeg in code.symmetry:
                            tmp = copy.deepcopy(top)
                            tmp.trace(leg[0],leg[1],Tensor(code,i), tractLeg)
                            queue.append(tmp)
            if len(top.tensorList)>=2 and top.selfTraceCount<=selfTraceDepth and top.get_n()>4:
                dangleLegs = top.equiv_trace_leg()
                for i in range(len(dangleLegs)-1):
                    for j in range(i+1, len(dangleLegs)):
                        first,second = dangleLegs[i],dangleLegs[j]
                        if first[0]==second[0]:
                            continue
                        if top.largerSelfTrace(('self',first[0],first[1],second[0],second[1])):
                            continue
                        tmp = copy.deepcopy(top)
                        tmp.selfTrace(first[0],first[1],second[0],second[1])
                        queue.append(tmp)
    end = time.time()
    print(f"use {end-start}s")
    return minError
            

if __name__ == "__main__":
    px = 0.01
    pz = 0.05
    import time
    start = time.time()
    minE = search(Tensor(code603, 0))
    print(minE)

    # t604 = Tensor(code604)
    # tn = TNNetwork(Tensor(code603))
    # tn.trace(0,2,Tensor(code604),0)
    # tn.trace(0,4,Tensor(code604),0)
    # tn.trace(0,3,Tensor(code604),0)
    # tn.selfTrace(2,1,3,1)
    # logLeg = tn.equiv_trace_leg()[0]
    # tn.setLogical(logLeg[0], logLeg[1])

    # insList = [['trace', 0, 2, Tensor(code603), 2], ['trace', 0, 1, Tensor(code604), 0], ['self', 0, 5, 2, 1]]
    # insList.append(("setLog", 0, 0))
    # print(eval_prog(insList, Tensor(code603)))
    # exit(0)
    

    # insList = [['trace', 0, 2, tn, 0], ['trace', 0, 4, tn, 0], ['trace', 0, 3, tn, 0], ['self', 2, 1, 3, 1]]
    # insList.append(("setLog", 0, 0))
    # eval_prog(insList, Tensor(code603))
    # print(eval_tn(tn))
    end = time.time()
    print(end-start)

