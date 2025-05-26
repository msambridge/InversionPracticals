import numpy as np
import matplotlib.pyplot as plt

def shaw(n,return_moddat=False):
    '''
    SHAW Test problem: one-dimensional image restoration model. 
    
    A,b,x = shaw(n) 
 
 Discretization of a first kind Fredholm integral equation with 
 [-pi/2,pi/2] as both integration intervals.  The kernel K and 
 the solution f, which are given by 
    K(s,t) = (cos(s) + cos(t))*(sin(u)/u)^2 
    u = pi*(sin(s) + sin(t)) 
    f(t) = a1*exp(-c1*(t - t1)^2) + a2*exp(-c2*(t - t2)^2) , 
 are discretized by simple quadrature to produce A and x. 
 Then the discrete right-hand b side is produced as b = A*x. 
 
 The order n must be even. 
 
 Reference: C. B. Shaw, Jr., "Improvements of the resolution of 
 an instrument by numerical solution of an integral equation", 
 J. Math. Anal. Appl. 37 (1972), 83-112. 
 
 Modified from the original by Richard Skelton, 2013, ANU.
 '''
    # Check input.
    if n % 2 != 0:
        raise ValueError('The order n must be even')

    # Initialization.
    h = np.pi / n
    G = np.zeros((n, n))

    # compute the matrix G
    vec = np.arange(.5,n+.5,1)

    co = np.cos(-np.pi/2+vec*h)
    psi = np.pi*np.sin(-np.pi/2+vec*h)
    for i in range(1,int(n/2+1)):
    	for j in range(i,n-i+1):
    		ss = psi[i-1]+psi[j-1]
    		G[i-1,j-1] = np.square((co[i-1]+co[j-1])*np.sin(ss)/float(ss))
    		G[n-j,n-i] = G[i-1,j-1]
    	G[i-1,n-i] = np.square(2*co[i-1])
    #G = S.array(G)
    G = G + np.triu(G,1).T; G = G*h

    if(return_moddat):
        # compute the vectors x and b
        a1 = 2; c1 = 6; t1 = 0.8
        a2 = 1; c2 = 2; t2 = -0.5
        x = a1*np.exp(-c1*(-np.pi/2 + vec*h -t1)**2) + a2*np.exp(-c2*(-np.pi/2 + vec*h-t2)**2)
        b = S.dot(G,x)
        return G, b, x
    else:
        return G

def pseudo_inverse(G,p): # calculate the pseudo-inverse with first p singular values
    u,s,v = np.linalg.svd(G) # singular value decomposition 
    # calculate the solution using the supplied rank of the generalised inverse

    Vp = v.T[:,0:p]
    Up = u[:,0:p].T
    Sp = np.diag(s[0:p])

    Gdagger = np.dot(Vp,np.dot(np.linalg.inv(Sp),Up)) # pseudo inverse
    return Gdagger