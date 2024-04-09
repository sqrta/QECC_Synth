from adt import *
import itertools
from enumerator import *
import traceback
import sys
import pickle 
import os
from collections import deque as Queue

save_dir = 'save/'

def eval_prog(prog, initial, px=0.01, pz = 0.05):
    tn = buildProg(prog, initial)
    tn.show()
    return eval_tn(tn)
 
def chooseProg(setlog, minError, f, px, pz, write=True):
    n = setlog.get_n()
    k = setlog.get_k()
    try:
        d, error, KS = eval_TN(setlog, px, pz)
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
                print(content)
                f.write(content+"\n\n")
    return d, error, KS

def dump(variable, fileName):
    with open(save_dir+fileName+'.pkl', 'wb') as f:
        pickle.dump(variable, f)

def load(fileName):
    with open(save_dir+fileName+'.pkl', 'rb') as f:
        tmp = pickle.load(f)
    return tmp

def save_back(filename):
    path = 'save_back/'
    source = 'save/'
    if isinstance(filename, list):
        for name in filename:          
            os.system(f'cp {source}{name}.pkl {path}')
    else:
        os.system(f'cp {source}{filename}.pkl {path}')

def search(initial, candidate_code, candidate_bound, px, pz, resume = False):
    import time
    start = time.time()
    queue = Queue()
    queue.append(TNNetwork(initial))
    nextQueue = Queue()
    maxTensor = sum(candidate_bound)
    selfTraceDepth = 4
    exist_set = set()
    minError = {}
    count = 0
    maxSize = 16
    MAX_QUEUE = 1e6
    MAX_ITER = 7e5
    queueCount = 0
    saveCount=0
    firstFill = False
    if resume:
        queue = load('queue')
        exist_set = load('exist_set')
        minError = load('minError')
        count = load('count')
        saveCount = load('saveCount')
        queueCount = load('queueCount')
    prefix = f"sfound"
    f = open(prefix+str(count), 'w')    
    print("Search Start")
    
    while MAX_ITER>0:
        count+=1
        MAX_ITER-=1
        if len(queue)>MAX_QUEUE:
            firstFill = True
        if len(nextQueue)>MAX_QUEUE:
            dump(nextQueue, f"Queue{saveCount}")
            saveCount+=1
            nextQueue.clear()
        if len(queue) == 0:
            queue = load(f'Queue{queueCount}')
            queueCount += 1
        
        # print(minError)
        # print(f"count: {count}, queue size: {sys.getsizeof(queue)}, dict size: {sys.getsizeof(exist_set)}")
        if count%5e4==0:
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
            dump(saveCount, 'saveCount')
            dump(queueCount, 'queueCount')   
        if count %1e5 ==0:
            save_back(['queue', 'minError', 'count', 'saveCount', 'queueCount', 'exist_set'])
        top = queue.popleft()

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
        #         chooseProg(setlog, minError, f, px, pz, write=True)

        hash = copy.deepcopy(top)
        hash.setLogical(logLeg[0], logLeg[1])
        d, error, Ks = chooseProg(hash, minError, f, px, pz, write=True)
        
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
                                if not firstFill:
                                    queue.append(tmp)
                                else:
                                    nextQueue.append(tmp)
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
                        if not firstFill:
                            queue.append(tmp)
                        else:
                            nextQueue.append(tmp)
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
    px = 0.05
    pz = 0.01
    import time
    start = time.time()
    candidate_code = ['code604', 'codeGHZ', 'code804','code603', 'codeH', 'codeS']
    resumation = False
    if len(sys.argv)>=2 and sys.argv[1]=='1':
        resumation = True
    print(resumation, sys.argv)
    minE = search(Tensor('code604', 0), candidate_code, [2,2, 1,2, 2,2], px, pz, resume = resumation)
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

