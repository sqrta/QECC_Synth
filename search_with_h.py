from adt import *
import itertools
from enumerator import *
import traceback
import sys
import pickle 
import random
import os

save_dir = 'save/'

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
            print(f"write, d: {d}")
            cm = setlog.toCm()
            rowW = cm.rowWBound()
            colW = cm.colWBound()
            content = str(setlog) + f"error: {error}, n: {n}, k: {k}, d: {d}, KS:{KS}, rowW: {rowW}, colW: {colW}"
            key = (n,d,rowW,colW)
            if key not in minError.keys() or error < minError[key]:
                minError[key] = error
                # print(content)
                f.write(content+"\n\n")
    return d, error, KS

def dump(variable, fileName):
    with open(save_dir+fileName+'.pkl', 'wb') as f:
        pickle.dump(variable, f)

def load(fileName):
    with open(save_dir+fileName+'.pkl', 'rb') as f:
        tmp = pickle.load(f)
    return tmp

def priority(edge) -> float:
    score = 0
    if (edge[0] in [0, 1] and edge[2] in [0, 1]) or \
            ((edge[0] == 2 and edge[2] == 3) or (edge[0] == 3 and edge[2] == 2)) or \
            ((edge[0] == 4 and edge[2] == 5) or (edge[0] == 5 and edge[2] == 4)):
        score = -1.5 if edge[1] * edge[3] == 0 else -1.3
    else:
        score = edge[0] * 0.01 + edge[1] * 0.02 - edge[2] * 0.002 - edge[3] ** 0.5
    
    if edge[1] - edge[3] == 0:
        score += 1.2
    elif edge[1] * edge[3] == 0:
        score += 0.6
    elif edge[0] in [0, 1]:
        score += 0.2
    elif edge[0] in [2, 3]:
        score += 0.4
    else:
        score += 0.6
    
    return score

def sort_with_h(edges, exist):
    result = sorted(edges, key=lambda edge: priority((exist[edge[0]], edge[1], edge[2], edge[3])), reverse=True)
    return result

def search(initial, candidate_code, candidate_bound, resume = False):
    import time
    start = time.time()
    queue = [TNNetwork(initial)]
    maxTensor = sum(candidate_bound)
    selfTraceDepth = 4
    exist_set = set()
    minError = {}
    count = 0
    maxSize = 14
    MAX_QUEUE = 6e6
    MAX_ITER = 7e5
    if resume:
        queue = load('queue')
        exist_set = load('exist_set')
        minError = load('minError')
        count = load('count')
    prefix = f"sfound"
    f = open(prefix+str(count), 'w')    
    while len(queue)>0 and MAX_ITER>0:
        count+=1
        MAX_ITER-=1
        # print(minError)
        # print(f"count: {count}, queue size: {sys.getsizeof(queue)}, dict size: {sys.getsizeof(exist_set)}")
        if count%5000==0:
            f.write(str(minError))
            end = time.time()
            f.write(f"\nqueue length: {len(queue)}\nuse {end-start}s\n")
            f.write(f"count: {count}, queue size: {sys.getsizeof(queue)}, dict size: {sys.getsizeof(exist_set)}\n")
            f.close()
            f = open(f"{prefix}{count}",'w')
            dump(queue, 'queue')
            dump(exist_set, 'exist_set')
            dump(minError, 'minError')
            dump(count, 'count')  
        if count %2e4 ==0:
            os.system('cp -r save save_back')             
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
        d, error, Ks = chooseProg(hash, minError, f, write=True)      
        if len(queue) < MAX_QUEUE and (d,error) not in exist_set:
            # print(f"count: {count}, d: {d}")
            exist_set.add((d,error))
            exist = [a.index for a in top.tensorList] 
            if top.get_n()<=maxSize-2 and len(top.tensorList)<maxTensor and (len(top.insList)<1 or top.insList[-1][0]!="self"):
                dangleLegs = top.equiv_trace_leg()              
                candidate_legs = []
                for i in range(len(candidate_code)):
                    codeName = candidate_code[i]
                    code = eval(codeName)
                    # if i< top.tensorList[-1].index:
                    #     continue
                    if exist.count(i) >= candidate_bound[i]:
                        continue
                    for leg in dangleLegs:
                        for tractLeg in code.symmetry:
                            candidate_legs.append((leg[0],leg[1],i, tractLeg))
                candidate_legs = sort_with_h(candidate_legs, exist)
                for leg in candidate_legs:
                    new_tnn = copy.deepcopy(top)
                    new_tnn.trace(leg[0],leg[1],Tensor(candidate_code[leg[2]],leg[2]), leg[3])
                    if new_tnn.get_n()<=maxSize:
                        queue.append(new_tnn)
            if len(top.tensorList)>=2 and top.selfTraceCount<=selfTraceDepth and top.get_n()>6:
                dangleLegs = top.equiv_trace_leg()
                candidate_legs = []
                for i in range(len(dangleLegs)-1):
                    for j in range(i+1, len(dangleLegs)):
                        first,second = dangleLegs[i],dangleLegs[j]
                        if first[0]==second[0]:
                            continue
                        if top.largerSelfTrace(('self',first[0],first[1],second[0],second[1])):
                            continue
                        candidate_legs.append((first[0],first[1],second[0],second[1]))
                candidate_legs = sort_with_h(candidate_legs, exist)
                for leg in candidate_legs:
                    new_tnn = copy.deepcopy(top)
                    new_tnn.selfTrace(leg[0],leg[1],leg[2],leg[3])
                    queue.append(new_tnn)
    end = time.time()
    print(f"use {end-start}s, count: {count}, iter_remain: {MAX_ITER}, queue length: {len(queue)}")
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

def GetTensorNetworkFromEdges(edges, tnList, max_legs):
    tracted = set()
    tracted_leg = set()
    new_edges = []
    insList = []
    def insert_edge(name, edge):
        insList.append([name] + list(edge))
        tracted.add(edge[0])
        tracted.add(edge[2])
        tracted_leg.add((edge[0], edge[1]))
        tracted_leg.add((edge[2], edge[3]))
    insert_edge('trace', edges.pop(0))
    index = 0
    while len(insList)<len(tnList)-1:
        edge = edges[index]
        t1 = edge[0]
        t2 = edge[2]
        if (t1, edge[1]) in tracted_leg or (t2, edge[3]) in tracted_leg:
            index += 1
            continue
        if (t1 in tracted and t2 in tracted) or (t1 not in tracted and t2 not in tracted):
            new_edges.append(edge)
        else:
            insert_edge('trace', edge)
        index += 1
    new_edges += edges[index:]
    to_add = max_legs - len(insList)
    for edge in new_edges:
        if len(insList) >= max_legs:
            break
        if (edge[0], edge[1]) not in tracted_leg and (edge[2], edge[3]) not in tracted_leg:
           insert_edge('self', edge)
    print(insList)
    print(tnList)
    insList, tnList = normal_prog(insList, tnList)    
    tn = GenProg2TNN(insList, tnList)
    logicalLeg = tn.equiv_trace_leg()[0]
    tn.setLogical(logicalLeg[0], logicalLeg[1])
    return tn
            

if __name__ == "__main__":
    px = 0.01
    pz = 0.05
    import time
    start = time.time()
    candidate_code = ['code804','code603', 'codeH', 'codeS', 'code604', 'codeGHZ']
    resumation = False
    if len(sys.argv)>=2 and sys.argv[1]=='1':
        resumation = True
    minE = search(Tensor('code603', 0), candidate_code, candidate_bound=[1,2, 2,2, 1,2],resume = resumation)
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

