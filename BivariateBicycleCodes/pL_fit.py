from math import exp
from scipy.optimize import curve_fit, fsolve
import numpy as np
from numpy import log as ln

def fit(dc, xdata, ydata):
    def func(x, c0, c1, c2):
        return dc/2 * ln(x) + c0 + c1*x + c2*x*x
    
    para, pcov = curve_fit(func, xdata, ydata)
    return para

def get_thres(dc, k, para):
    c0, c1, c2 = para[0], para[1], para[2]
    def func(x):
        return dc/2 * ln(x) + c0 + c1*x + c2*x*x - ln(k*x)

    pthres = fsolve(func, 0.01)
    return pthres

if __name__ == '__main__':
    xdata = [0.002, 0.003, 0.004, 0.005]
    ydata = [(17+5+21+19)/4e5, (134+128)/1e5, 277/1e4, 235/1776.0]
    d = 10
    dc = 8
    k = 16

    # ydata = [16/1e5, 449/1e5, (179 + 185) / 1e4, 0.01*12]
    # d = 12
    # dc = 10
    # k = 12
    ydata = [ln(t / d) for t in ydata]
    para = fit(dc, np.array(xdata), np.array(ydata))
    print(para)
    pthres = get_thres(dc, k, para)
    print(pthres)