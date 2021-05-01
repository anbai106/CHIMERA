"""
###########################################################################
# @file optimization_utils.py
# @brief Functions for optimizing transformations and parameters.
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

def initialization(x,y,K):
    np.random.seed()
    D,M = x.shape
    N = y.shape[1]
    params = {'delta':None,'sigsq':None,'T':None,'t':None}    

    params['delta'] = np.ones((K,M))/K
    sigsq = 0
    for n in range(N):
        tmp = x - y[:,n].reshape(-1,1)
        sigsq = sigsq + np.sum(np.power(tmp,2))
    params['sigsq'] = sigsq/D/M/N;
    params['T'] = np.repeat(np.eye(D).reshape(D,D,1),K,axis=2)
    params['t'] = np.random.uniform(size=(D,K))
    return params

def transform( x,params ):
    T = params['T']
    t = params['t']
    delta = params['delta']
    
    [D,M] = x.shape
    K = T.shape[2]
    transformed_x = np.zeros((D,M))
    
    Tym = np.zeros((D,M,K))
    for k in range(K):
        Tym[:,:,k] = np.dot(T[:,:,k], x) + t[:,k].reshape(-1,1)
    
    for m in range(M):
        tmp = np.zeros(D)
        for k in range(K):
            tmp = tmp + delta[k,m] * Tym[:,m,k]
        transformed_x[:,m] = tmp;

    return transformed_x
    
def transform2( x,params ):
    T = params['T']
    delta = params['delta']
    
    D,M = x.shape
    K = T.shape[2]
    transformed_x = np.zeros((D,M))
    
    Tym = np.zeros((D,M,K))
    for k in range(K):
        Tym[:,:,k] = np.dot(T[:,:,k],x)
    
    for m in range(M):
        tmp = np.zeros(D)
        for k in range(K):
            tmp = tmp + delta[k,m] * Tym[:,m,k]
        transformed_x[:,m] = tmp
    return transformed_x

def transform3( x,params ):
    T = params['T']
    t = params['t']
    
    D,K = t.shape
    transformed_x = np.zeros((D,K))
    
    for k in range(K):
        transformed_x[:,k] = np.dot(T[:,:,k],x) + t[:,k]
    return transformed_x

def Estep(y,yd,ys,tx,xd,xs,sigsq,r,rs):
    """Expectation calculation.
    """
    M = tx.shape[1]
    N = y.shape[1]

    #> calculate RBF kernel distance based on imaging features
    D1 = np.diag(np.dot(Tr(y),y))
    D2 = np.diag(np.dot(Tr(tx),tx))
    Mid = 2 * np.dot(Tr(y),tx)
    tmp1 = D1.reshape(-1,1).repeat(M,axis=1) - Mid + D2.reshape(1,-1).repeat(N,axis=0)
    
    #> calculate RBF kernel distance based on covariate features
    tmp2 = np.zeros(tmp1.shape)
    if r != 0:
        D1 = np.diag(np.dot(Tr(yd),yd))
        D2 = np.diag(np.dot(Tr(xd),xd))
        Mid = 2 * np.dot(Tr(yd),xd)
        tmp2 = D1.reshape(-1,1).repeat(M,axis=1) - Mid + D2.reshape(1,-1).repeat(N,axis=0)
    
    #> calculate RBF kernel distance based on set information
    tmp3 = np.zeros(tmp1.shape)
    if rs != 0:
        D1 = np.diag(np.dot(Tr(ys),ys))
        D2 = np.diag(np.dot(Tr(xs),xs))
        Mid = 2 * np.dot(Tr(ys),xs)
        tmp3 = D1.reshape(-1,1).repeat(M,axis=1) - Mid + D2.reshape(1,-1).repeat(N,axis=0)
    
    #> combine distances and normlize to probability distribution
    P = np.exp((-tmp1-r*tmp2-rs*tmp3)/2/sigsq)+np.finfo(np.float).tiny
    P = P/np.sum(P,axis=1).reshape(-1,1)
    
    return P


def Mstep(y,yd,ys,x,tx,xd,xs,P,params,config):
    """Mstep optimization, for different transformation import different modules
    """
    if config['transform'] == 'affine':
        from .Mstep_affine import solve_sigsq,solve_delta,solve_T,solve_t
    elif config['transform'] == 'duo':
        from .Mstep_duo import solve_sigsq,solve_delta,solve_T,solve_t
    else:
        from .Mstep_trans import solve_sigsq,solve_delta,solve_T,solve_t
    
    params['sigsq'] = solve_sigsq(y,yd,ys,tx,xd,xs,P,params,config)
    params['delta'] = solve_delta(y,x,P,params)
    params['T'] = solve_T(y,x,P,params,config)
    params['t'] = solve_t(y,x,P,params,config)
    
    return params
    
def calc_obj(x,y,xd,yd,xs,ys,P,params,config):
    """Objective function calculation
    """
    lambda1 = config['lambda1']
    lambda2 = config['lambda2']
    r  = config['r']
    rs = config['rs']
    K  = config['K']
    
    D,N = y.shape
    M   = x.shape[1]
    d   = 0
    ds  = 0
    
    IM = np.ones((M,1))
    IN = np.ones((N,1))
    
    tx = transform(x,params)
    tmp = 0
    for i in range(K):
        tmp = tmp + np.power(np.linalg.norm(params['T'][:,:,i]-np.eye(D),'fro'),2)
        
    P1 = np.diag(np.dot(P,IM).flatten())
    P2 = np.diag(np.dot(Tr(P),IN).flatten())

    term1 = np.trace(y.dot(P1).dot(Tr(y)) - 2*y.dot(P).dot(Tr(tx)) + tx.dot(P2).dot(Tr(tx)))
    term2 = 0
    if r != 0:
        d = xd.shape[0]
        term2 = r * np.trace(yd.dot(P1).dot(Tr(yd)) - 2*yd.dot(P).dot(Tr(xd)) + xd.dot(P2).dot(Tr(xd)))
    term3 = 0
    if rs != 0:
        ds = 1
        term3 = rs * np.trace(ys.dot(P1).dot(Tr(ys)) - 2*ys.dot(P).dot(Tr(xs)) + xs.dot(P2).dot(Tr(xs)))
    obj = 0.5/params['sigsq'] * ( term1 + term2 + term3 \
           + lambda1*np.power(np.linalg.norm(params['t'],'fro'),2) +lambda2*tmp) \
           + N*(D+d+ds)/2.0*np.log(params['sigsq'])

    return obj   
