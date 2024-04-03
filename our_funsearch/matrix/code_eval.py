from ibm_eval import *

def getMat(p, l, m):
    M = get_x(l, m) if p[0]=='x' else get_y(l, m)
    return mp(M, int(p[1]))

def string2Matrix(string, l, m):
    mList = [getMat(p,l,m) for p in string.split('+')]
    return sumMat(mList)


if __name__ == '__main__':
    gap = start(['/mnt/e/github/tmp/gap-4.13.0/gap', '-L', 'workplace','-q', '-b'])
    l = 3
    m = 10
    Mstring = "x0+x1+y1+y9, x0+x2+y3+y7"
    M2gba = Mstring.split(', ')
    A = string2Matrix(M2gba[0], l, m)
    B = string2Matrix(M2gba[1], l, m)
    k, d = Get_kd_BBCode(gap, A, B, l, m)
    print(k,d)