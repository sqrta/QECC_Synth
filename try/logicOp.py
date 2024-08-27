from Hamil_search import *

COMMUTE = 0
MULTI = 1

def Cons2Str(constraint, PauliSet):
    ind = constraint.consIndex
    type = constraint.type
    eff = constraint.eff
    if type == COMMUTE:
        return f'{eff} commute with {PauliSet[ind[0]]}'
    elif type == MULTI:
        return f'{eff} product of {PauliSet[ind[0]]} and {PauliSet[ind[1]]}'
    return ""

class Cons:
    def __init__(self, consIndex, type, eff=1) -> None:
        self.consIndex = consIndex
        self.type = type
        self.eff= eff

    def __str__(self) -> str:
        return f'{self.eff}*{self.term}'

    def __repr__(self):
        return str(self)

def searchLogical(n,k,H):
    P = getProjector(n,H)
    phyCanStr = phyOpCandiate(n)
    phyCan = [pauliExpr2Mat(n, i) for i in phyCanStr]
    '''
    ConstraintSet, the first op is free so is []
    rules for each op is a list of CONS)
    '''
    '''
        order 
    
    '''
    ConstraintSet = []
    index = [(i,j) for i in range(k-1) for j in range(i+1, k)]
    PauliSet = [f'X{i}' for i in range(k)] + [f'Z{i}' for i in range(k)]
    PauliSet += [f'X{a[0]}X{a[1]}' for a in index] + [f'Z{a[0]}Z{a[1]}' for a in index]
    # add X0~X_{k-1}
    for i in range(k):
        rule = []
        for j in range(i):
            rule.append(Cons([j], COMMUTE, 1))
        ConstraintSet.append(rule)
    # Z0~Z_{n-1}
    for i in range(k):
        rule = []
        for j in range(k):
            if i!=j:
                rule.append(Cons([j], COMMUTE, 1))
            else:
                rule.append(Cons([j], COMMUTE, -1))
        for j in range(i):
            rule.append(Cons([k+j], COMMUTE, 1))   
        ConstraintSet.append(rule)  
    # add X0X1 ...
    for ind in index:
        a,b = ind        
        rule=[Cons([a,b], MULTI, 1)]
        ConstraintSet.append(rule)

    # add Z0Z1 ...
    for ind in index:
        a,b = ind        
        rule=[Cons([a+k,b+k], MULTI, 1)]
        ConstraintSet.append(rule)

    for i in range(len(ConstraintSet)):
        cs = ConstraintSet[i]
        tmp = [Cons2Str(c, PauliSet) for c in cs]
        print(f'{PauliSet[i]}: {", ".join(tmp)}')

    indexStack = [0]
    print(f"conSetLen: {len(ConstraintSet)}, candiateLen: {len(phyCan)}")

    while len(indexStack)>0:
        print(indexStack)
        if len(indexStack)>=len(ConstraintSet):
            break
        curLog = len(indexStack)-1
        curPhyInd = indexStack[curLog]
        if curPhyInd >= len(phyCan):
            indexStack.pop(-1)
            indexStack[-1] += 1
            continue
        phyOp = phyCan[curPhyInd]
        flag = True
        for cons in ConstraintSet[curLog]:
            relatedOp = [phyCan[i] for i in cons.consIndex]
            if not testConstraint(P, phyOp, cons, relatedOp):
                flag = False
                break
        if flag:
            if len(indexStack) == len(ConstraintSet):
                break
            else:
                indexStack.append(0)
        else:
            indexStack[-1] += 1

    return PauliSet, [phyCanStr[i] for i in indexStack]

def testConstraint(P, phyOp, cons, relatedOp):
    type = cons.type
    eff = cons.eff
    if type == COMMUTE:
        return commuteOrNot(P@phyOp, P@relatedOp[0], eff)
    if type == MULTI:
        return checkSame(P@phyOp, P@relatedOp[0] @ P@ relatedOp[1])
    return False

if __name__ =='__main__':
    n = 6
    k = 3
    Xeff, Zeff = (0, -1, -2, -1, 0, 1), (2, 0, -2, 0, 2, 1)
    H = getHamil(n,Xeff,Zeff)
    list, res = searchLogical(n, k, H)
    print(list)
    print(res)