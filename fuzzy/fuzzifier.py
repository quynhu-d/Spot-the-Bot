import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from tqdm.notebook import tqdm, trange
from pynverse import inversefunc    # to get inverse function
from scipy.integrate import quad    # for integrating
from numpy.linalg import norm

def mu_func(mus, xs, ps=None, dc=3, l=2, r=2, leave_verbose=False):
    m1 = np.zeros(xs.shape)
    m2 = np.zeros(xs.shape)
    ps = np.random.binomial(n=1,size=xs.shape, p=0.5)    # probability for left/right slope
    dim = xs.shape[1]
    for i, (x, mu, p) in tqdm(enumerate(zip(xs, mus, ps)), leave=leave_verbose, 
                              desc='Fuzzifying...', total=xs.shape[0]):
        symbols = []
        sols = []
        eq = 1
        for pj, xj in zip(p, x):
            m = Symbol('x')
            if pj == 1:
                sols.append(solve((1 - (m - xj) / l) - np.power(mu,1/dim), m)[0])
            else:
                sols.append(solve((1 - (xj - m) / r) - np.power(mu,1/dim), m)[0])
        for j in range(dim):
            if p[j] == 1:
                m1[i][j] = sols[j]
                m2[i][j] = m1[i][j] + dc
            else:
                m2[i][j] = sols[j]
                m1[i][j] = m2[i][j] - dc
    return m1, m2, ps

def mu_func_2(mus, xs, ps=None, dc=3, l=2, r=2, leave_verbose=False):
    m1 = np.zeros(xs.shape)
    m2 = np.zeros(xs.shape)
    if ps is None:
        ps = np.random.binomial(n=1,size=xs.shape, p=0.5)    # probability for left/right slope
    dim = xs.shape[1]
    for i, (x, mu, p) in tqdm(enumerate(zip(xs, mus, ps)), desc='Fuzzifying...', 
                              leave=leave_verbose, total=xs.shape[0]):
        m = (l * p + (1-p) * r) * (1-np.power(mu, 1/dim)) * (2*p-1) + x
        m1[i][p == 1] = m[p == 1]
        m2[i][p == 0] = m[p == 0]
        m1[i][p == 0] = m2[i][p == 0] - dc
        m2[i][p == 1] = m1[i][p == 1] + dc
    return m1, m2, ps

def draw_trapmf(m1, m2, axs, dc=3, l=2,r=2, x=None, mu=None):
    axs.plot([m1,m2], np.ones(2))
    axs.plot([m1-l,m1],[0,1])
    axs.plot([m2,m2+r], [1,0])
    axs.plot([x],[mu],'x')
    return axs

def fuzzy_distance(x_1, x_2, lambda_=.5, ro_=.5, L=None, R=None):    # default lambda = ro = .5 - trapezoidal mfunc
    m1_1, m2_1, l_1, r_1 = x_1
    m1_2, m2_2, l_2, r_2 = x_2
    if lambda_ is None:
        if L != None:
            L_inv = inversefunc(L)
            lambda_ = quad(L_inv, 0, 1)[0]
        else:
            raise NameError('No information about L-side')
    if ro_ is None:
        if R != None:
            R_inv = inversefunc(R)
            ro_ = quad(R_inv, 0, 1)[0]
        else:
            raise NameError('No information about R-side')

    return np.sqrt(
        norm(m1_1 - m1_2) ** 2 + norm(m2_1 - m2_2) ** 2 + \
        norm((m1_1 - lambda_ * l_1) - (m1_2 - lambda_ * l_2)) ** 2 +\
        norm((m2_1 - ro_ * r_1) - (m2_2 - ro_ * r_2)) ** 2
    )

class Fuzzifier:
    def __init__(self, l, r, dc):
        self.l = l    # length of left slope
        self.r = r    # length of right slope
        self.dc = dc    # length for mu == 1

    def fuzzify(self, X, mus, verbose=False):
        m1, m2, p = mu_func_2(mus, X, l=self.l, r=self.r, dc=self.dc, leave_verbose=verbose)
        return np.array(list(zip(m1, m2, np.full(X.shape[0], self.l), np.full(X.shape[0], self.r))), dtype=object)
