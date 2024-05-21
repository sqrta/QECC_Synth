from Hamil_search import *

n=4
vec = ket2Vec(n, ['0'*n])
Xeff = (1,2,1,2)
Zeff = (1,-3,1,-3)
terms = [PauliTerm(n, f'X{i}*X{(i+1)%n}', Xeff[i]) for i in range(n)] 
terms += [PauliTerm(n, f'Z{i}*Z{(i+1)%n}', Zeff[i]) for i in range(n)]
# print(terms)
H = sum([t.value() for t in terms])
print(testH(n, H))
# print(H @ vec)