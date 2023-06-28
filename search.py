from adt import *
import itertools
from enumerator import *

def buildProg(insList, initial):
    tn = TNNetwork(initial)
    for ins in insList:
        if ins[0] == 'trace':
            tn.trace(ins[1], ins[2], ins[3], ins[4])
        elif ins[0] == "self":
            tn.selfTrace(ins[1], ins[2], ins[3], ins[4])
        elif ins[0] == "setLog":
            tn.setLogical(ins[1], ins[2])
    return tn

def eval_prog(prog, initial, px=0.01, pz = 0.05):
    tn = buildProg(prog, initial)
    tn.show()
    return eval_tn(tn)

def eval_tn(tn):
    n = tn.get_n()
    k = tn.get_k()

    APoly = parse(tn)
    return distance_from_poly(simp_poly(APoly), n, k), xzNoise(n, k, APoly, px, pz) 

def search(initial):
    import time
    start = time.time()
    queue = [TNNetwork(initial)]
    maxTensor = 3
    selfTraceDepth = 2
    candidate_code = [code603, code604]
    exist_set = set()
    minError = 1 
    maxD = 3
    with open("found", 'w') as f:
        while len(queue)>0:
            top = queue.pop(0)
            logLeg = top.equiv_trace_leg()[0]
            setlog = copy.deepcopy(top)
            setlog.setLogical(logLeg[0], logLeg[1])
            d, error = eval_tn(setlog)
            if d>=3:
                content = str(setlog) + f"error: {error}, d: {d}"
                print(content)
                if error < minError or d>3:
                    minError = error
                    print(f"d: {d}, current MinE: {minError}\n")
                    f.write(content+"\n\n")
            if (d,error) not in exist_set:
                exist_set.add((d,error))
                if len(top.tensorList)<maxTensor and (len(top.insList)<1 or top.insList[-1][0]!="self"):
                    dangleLegs = top.equiv_trace_leg()
                    for code in candidate_code:
                        for leg in dangleLegs:
                            for tractLeg in code.symmetry:
                                tmp = copy.deepcopy(top)
                                tmp.trace(leg[0],leg[1],Tensor(code), tractLeg)
                                queue.append(tmp)
                if len(top.tensorList)>=2 and top.selfTraceCount<=selfTraceDepth:
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
    # insList = [['trace', 0, 2, Tensor(code603), 2], ['trace', 0, 1, Tensor(code604), 0], ['self', 0, 5, 2, 1]]
    # insList.append(("setLog", 0, 0))
    # print(eval_prog(insList, Tensor(code603)))
    # exit(0)
    minE = search(Tensor(code603))
    print(minE)

