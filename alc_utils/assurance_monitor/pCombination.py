import numpy as np
from scipy.stats import norm, combine_pvalues, cauchy
from scipy.special import comb, betainc
import math


def ECF(p_values):  # Extended Chi-square Function (Fisher's method)
    # print(p_values)
    t = np.prod(p_values, axis=0)
    # print(t)
    s = np.zeros(p_values.shape[1])
    for i in range(p_values.shape[0]):
        for j in range(p_values.shape[1]):
            if t[j] > 10e-10:
                s[j] += pow(-math.log(t[j]), i)/(1.0*math.factorial(i))
    return t*s

# def scipy_fisher(p_values):
#     res=np.empty(p_values.shape[1])
#     for i in range(p_values.shape[1]):
#         _,pval=combine_pvalues(p_values[:,i],method='fisher')
#         res[i]=pval
#     return res


def stouffer(p_values):  # Stouffer's Z-score method
    res = np.empty(p_values.shape[1])
    for i in range(p_values.shape[1]):
        _, pval = combine_pvalues(p_values[:, i], method='stouffer')
        res[i] = pval
    return res


def weighted_stouffer(p_values):
    a = np.arange(1, p_values.shape[0]+1)
    w = a/(1.0*np.sum(a))
    res = np.empty(p_values.shape[1])
    for i in range(p_values.shape[1]):
        _, pval = combine_pvalues(p_values[:, i], method='stouffer', weights=w)
        res[i] = pval
    return res


def mudholkar_george(p_values):
    res = np.empty(p_values.shape[1])
    for i in range(p_values.shape[1]):
        _, pval = combine_pvalues(p_values[:, i], method='mudholkar_george')
        res[i] = pval
    return res


def pearson(p_values):
    res = np.empty(p_values.shape[1])
    for i in range(p_values.shape[1]):
        _, pval = combine_pvalues(p_values[:, i], method='pearson')
        res[i] = pval
    return res


def cauchi(p_values):
    T = np.mean(np.tan((0.5-p_values)*np.pi), axis=0)
    return (0.5-(np.arctan(T)/np.pi))
    # return 1 - cauchy.cdf(T)


def irwin_hall(p_values):
    n = p_values.shape[0]
    t = np.empty(p_values.shape[1], dtype=np.int64)
    t = np.floor(np.sum(p_values, axis=0))
    res = np.zeros(p_values.shape[1])

    for i in range(p_values.shape[1]):
        for k in range(int(t[i])+1):
            res[i] += (pow(-1, k)*comb(n, k)*pow(int(t[i])-k, n))
        res[i] /= (1.0*math.factorial(n))
    return res


def min_cdf(p_values):
    n = p_values.shape[0]
    t = np.min(p_values, axis=0)
    return betainc(1, n, t)


# when ord=1, same as above. ord=2 second lowest and so on...
def ord_stat_cdf(p_values):
    ord = 2
    n = p_values.shape[0]
    if n < ord:
        ord = n
    t = np.min(p_values, axis=0)
    return betainc(ord, n-ord+1, t)


def SNF(p_values):
    q = norm.ppf(p_values)
    s = np.sum(q)
    return norm.cdf(s)


def BH(p_values):
    p_values.sort()

# Merging functions


def arith_avg(p_values):
    return np.mean(p_values, axis=0)


def geom_avg(p_values):
    return p_values.prod(axis=0)**(1.0/len(p_values))


def pmin(p_values):
    return np.min(p_values, axis=0)


def pmax(p_values):
    return np.max(p_values, axis=0)

# Arbitrary


def ecdf_sum(p_values):
    return p_values.sum(axis=0)


def ecdf_product(p_values):
    return p_values.prod(axis=0)


def ecdf_min(p_values):
    return p_values.min(axis=0)


def ecdf_max(p_values):
    return p_values.max(axis=0)
