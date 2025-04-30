###################
import numpy as np
def eqlocate(x0,y0,z0,ts,la,lo,el,vp,tol):
    la2km=111.19
    lo2km=75.82
    
    i=5-1
    t0=ts[i]-np.sqrt(((x0*lo2km-lo[i]*lo2km)**2)+((y0*la2km-la[i]*la2km)**2)\
    +(el[i]-z0)**2)/vp
    
    ni=0
    sols=[[t0,x0,y0,z0]]
    
    while 1:
        ni=ni+1
        D0=[]
        for i in range(25):
            D0.append(np.sqrt((lo[i]*lo2km-x0*lo2km)**2+(la[i]*la2km-y0*la2km)**2+(el[i]-z0)**2))
        G=[]
        d=[]
        for i in range(25):
            G.append([1,((x0-lo[i])*lo2km)/(D0[i]*vp),((y0-la[i])*la2km)/(D0[i]*vp)])
            d.append(ts[i]-(D0[i]/vp+t0))
        G=np.array(G)
        d=np.array(d)
        m=np.linalg.lstsq(G,d,rcond=None)[0]
        t0=t0+m[0]
        x0=x0+m[1]/lo2km
        y0=y0+m[2]/la2km
        
        sols.append([t0,x0,y0,z0])        
        if m[0]<tol[0] and np.sqrt(m[1]**2+m[2]**2)<tol[1]:
            break
    sols=np.array(sols)
    return sols, d