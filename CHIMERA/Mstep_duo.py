"""
###########################################################################
# @file Mstep_duo.py
# @brief Mstep optimization for transformation with diagonal A
#
# @author Aoyan Dong
#
# @Link: https://www.cbica.upenn.edu/sbia/software/
#
# @Contact: sbia-software@uphs.upenn.edu
##########################################################################
"""
import numpy as np
from numpy import transpose as Tr
from .optimization_utils import *
def solve_sigsq(y,yd,ys,tx,xd,xs,P,params,config):
    D,N = y.shape
    d = 0
    K,M = params['delta'].shape
    IM = np.ones((M,1))
    IN = np.ones((N,1))
    tmp = 0
    for i in range(K):
        tmp = tmp + np.power(np.linalg.norm(params['T'][:,:,i]-np.eye(D),'fro'),2)
    
    P1 = np.diag(np.dot(P,IM).flatten())
    P2 = np.diag(np.dot(Tr(P),IN).flatten())

    term1 = np.trace(y.dot(P1).dot(Tr(y)) - 2*y.dot(P).dot(Tr(tx)) + tx.dot(P2).dot(Tr(tx)))
    term2 = 0
    if config['r']!=0:
        d = yd.shape[0]
        term2 = config['r'] * np.trace(yd.dot(P1).dot(Tr(yd)) - 2*yd.dot(P).dot(Tr(xd)) + xd.dot(P2).dot(Tr(xd)))
    term3 = 0
    if config['rs']!=0:
        term3 = config['rs'] * (ys.dot(P1).dot(Tr(ys)) - 2*ys.dot(P).dot(Tr(xs)) + xs.dot(P2).dot(Tr(xs)))
    sigsq = 1.0/N/(D+d) * ( term1 + term2 + term3 + config['lambda1']*np.power(np.linalg.norm(params['t'],'fro'),2) + config['lambda2']*tmp)
    return sigsq

def solve_t(y,x,P,params,config):
    K = params['delta'].shape[0]
    I = np.eye(K)
    
    W,Z = prepare_t(y,x,P,params)
    
    t = Tr(np.linalg.solve(config['lambda1']*I + W, Z))
    return t
    
def prepare_t(y,x,P,params):
    D,N = y.shape
    delta = params['delta']
    K,M = delta.shape 
    
    IM = np.ones((M,1))
    IN = np.ones((N,1))
    
    P2 = np.sum(P,axis=0)
    W = np.zeros((K,K))
    Z = np.zeros((K,D))
    for m in range(M):
        for i in range(K):
            for j in range(i,K):
                W[i,j] = W[i,j] + P2[m]*delta[i,m]*delta[j,m]
    for i in range(1,K):
        for j in range(i):
            W[i,j] = W[j,i]
    
    x2 = transform2(x,params)
    
    for k in range(K):
        Z[k,:] = (y.dot(np.diag((delta[k,:].dot(Tr(P))).flatten())).dot(IN)\
                - x2.dot(np.diag(delta[k,:]*(Tr(P).dot(IN).flatten()))).dot(IM)).flatten()
    return W,Z


def solve_T(y,x,P,params,config):
    K = params['delta'].shape[0]
    D = x.shape[0]
    I = np.eye(K)
    IK = np.ones((K,1))
        
    W,U = prepare_T(y,x,P,params)

    A = np.zeros((K,D))
    for J in range(D):
        A[:,J] = np.linalg.solve((config['lambda2']*I + W[J]),\
                (config['lambda2']*IK - U[:,J].reshape(-1,1))).flatten()
        
    T = np.zeros((D,D,K))
    for i in range(K):
        T[:,:,i] = np.diag(A[i,:])
    
    return T

def prepare_T(y,x,P,params):
    D,N = y.shape
    delta = params['delta']
    K,M = delta.shape
    
    P2 = np.sum(P,axis=0)
    x2 = x*x
    W = np.zeros((D,K,K))
    
    for m in range(M):
        core_w = np.zeros((K,K))
        for i in range(K):
            for j in range(i,K):
                core_w[i,j] = P2[m]*delta[i,m]*delta[j,m]
                core_w[j,i] = core_w[i,j]
        for J in range(D):
            W[J] += core_w * x2[J,m]

    U1 = np.zeros((K,D))
    ZX = params['t'].dot(delta) * x
    for k in range(K):
        PD = (P2*delta[k,:]).reshape(1,-1)
        U1[k,:] = PD.dot(Tr(ZX))
    U2 = np.zeros((K,D))
    for k in range(K):
        U2[k,:] = np.sum((y.dot(P)*delta[k,:]) * x, axis=1)
    U = U1-U2
    return W,U

def solve_delta(y,x,P,params):
    K,M = params['delta'].shape
    tx = transform(x,params)
    delta = np.copy(params['delta'])
    
    P2 = np.sum(P,axis=0)
    for m in range(M):
        tx2 = transform3(x[:,m],params)
        tmp = y - tx[:,m].reshape(-1,1)
        d_delta = Tr(P[:,m]).dot(Tr(tmp)).dot(-tx2) / params['sigsq']

        Hm = 1.0/params['sigsq'] * P2[m] * (Tr(tx2)).dot(tx2)

        v = params['delta'][:,m] - np.linalg.inv(Hm + 0.001*np.eye(K)).dot(Tr(d_delta))
        delta[:,m] = project_simplex(v)

    return delta


def project_simplex(v):
    w = np.copy(v)
    if np.sum(np.isinf(v)) != 0:
        t1 = np.isinf(v) and (v>0)
        if np.sum(t1)!=0:
            w = np.zeros(len(v))
            w[np.nonzero(t1)[0]] = 1.0/np.sum(t1)
        else:
            t2 = np.isinf(v)
            if np.sum(t2) == len(t2):
                w[np.nonzero(t2)[0]] = 1.0/np.sum(t2)
            else:
                w[np.nonzero(t2)[0]] = 0
                w[np.nonzero(np.logical_not(t2))[0]] = 1.0/np.sum(np.logical_not(t2))
    else:
        mu = np.sort(v)[::-1]
        n = len(v)
        tmp = np.zeros(n)
        for j in range(n):
            tmp[j] = mu[j] - 1.0/(j+1) * (np.sum(mu[:(j+1)])-1)
        p = np.nonzero(tmp>=0)[0][-1]
        theta = 1.0/(p+1)*(np.sum(mu[:(p+1)])-1)
        for j in range(n):
            w[j] = max(v[j]-theta,0)
    return w
