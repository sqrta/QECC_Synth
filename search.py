from adt import *
import itertools
from enumerator import *
import traceback

def eval_prog(prog, initial, px=0.01, pz = 0.05):
    tn = buildProg(prog, initial)
    tn.show()
    return eval_tn(tn)
 
def chooseProg(setlog, minError, f, write=True):
    n = setlog.get_n()
    k = setlog.get_k()
    try:
        d, error, KS = eval_TN(setlog)
    except Exception: 
        print(str(setlog))
        traceback.print_exc()             
        print("error in eval!")
        exit(0)
    # if not d or d>=3 or KS!=1:
    if write:
        if d and d>=3:
            cm = setlog.toCm()
            rowW = cm.rowWBound()
            colW = cm.colWBound()
            content = str(setlog) + f"error: {error}, n: {n}, k: {k}, d: {d}, KS:{KS}, rowW: {rowW}, colW: {colW}"
            # print(content)
            key = (n,d,rowW,colW)
            if key not in minError.keys() or error < minError[key]:
                minError[key] = error
                # print(content)
                f.write(content+"\n\n")
    return d, error, KS


def search(initial, candidate_code, candidate_bound):
    import time
    start = time.time()
    queue = [TNNetwork(initial)]
    maxTensor = sum(candidate_bound)
    selfTraceDepth = 4
    exist_set = set()
    minError = {}
    count = 0
    prefix = "sfound"
    f = open(prefix, 'w')
    maxSize = 14
    while len(queue)>0:
        count+=1
        # print(count)
        if count%5000==0:
            f.write(str(minError))
            end = time.time()
            f.write(f"queue length: {len(queue)}\nuse {end-start}s")
            f.close()
            f= open(f"{prefix}{count}",'w')
        top = queue.pop(0)
        # logLeg = None
        # for leg in top.equiv_trace_leg():
        #     if leg[0]==0 and leg[1]>1:
        #         logLeg = leg
        #         break
        legs = top.equiv_trace_leg()
        logLeg = legs[0]        
        # if True:
        #     for secLeg in legs[1:]:
        #         setlog = copy.deepcopy(top)
        #         setlog.setLogical(logLeg[0], logLeg[1])
        #         setlog.setLogical(secLeg[0], secLeg[1])
        #         chooseProg(setlog, minError, f)

        hash = copy.deepcopy(top)
        hash.setLogical(logLeg[0], logLeg[1])
        d, error, Ks = chooseProg(hash, minError, f, write=False)
    
        if (d,error) not in exist_set:
            exist_set.add((d,error))
            if top.get_n()<=maxSize-2 and len(top.tensorList)<maxTensor and (len(top.insList)<1 or top.insList[-1][0]!="self"):
                dangleLegs = top.equiv_trace_leg()
                exist = [a.index for a in top.tensorList] 
                for i in range(len(candidate_code)):
                    codeName = candidate_code[i]
                    code = eval(codeName)
                    # if i< top.tensorList[-1].index:
                    #     continue
                    if exist.count(i) >= candidate_bound[i]:
                        continue
                    for leg in dangleLegs:
                        for tractLeg in code.symmetry:
                            tmp = copy.deepcopy(top)
                            tmp.trace(leg[0],leg[1],Tensor(codeName,i), tractLeg)
                            if tmp.get_n()<=maxSize:
                                queue.append(tmp)
            if len(top.tensorList)>=2 and top.selfTraceCount<=selfTraceDepth and top.get_n()>6:
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

def get_all_edge(tnList):
    legs = []
    tensor_list  = [eval(t) for t in tnList]
    for i in range(len(tensor_list)):
        for j in range(i+1, len(tensor_list)):
            for leg1 in range(tensor_list[i].length):
                for leg2 in range(tensor_list[j].length):
                    legs.append([i, leg1, j, leg2])
    return np.array(legs)
            

if __name__ == "__main__":
    px = 0.01
    pz = 0.05
    import time
    start = time.time()
    candidate_code = ['code804','code603', 'codeH', 'codeS', 'code604', 'codeGHZ']
    minE = search(Tensor('code603', 0), candidate_code, candidate_bound=[1,2, 2,2, 1,2])
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

