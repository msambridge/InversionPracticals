import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Ellipse

###################
# Data reading utilities
###################
def readeqdata(filename):
    with open(filename) as f:
        Lines=f.read().splitlines()  
     
    la, lo, el, ts =[],[],[],[]

    for i in range(len(Lines)):
        la.append(float(Lines[i].split()[0]))
        lo.append(float(Lines[i].split()[1]))
        el.append(float(Lines[i].split()[2]))
        ts.append(float(Lines[i].split()[3]))

    la,lo,el,ts =np.array(la), np.array(lo), np.array(el), np.array(ts)

    return la,lo,el,ts

def readborderdata(filename):
    # load border.xy
    border1, border2 =[], []

    with open(filename) as f1:
        border=f1.read().splitlines()  
    for i in range(len(border)):
        border1.append(float(border[i].split()[0]))
        border2.append(float(border[i].split()[1]))

    return np.array(border1),np.array(border2)

def eqlocate(x0,y0,z0,ts,la,lo,el,vpin,tol,solvedep=False,nimax=100,verbose=False,kms2deg=[111.19,75.82]):
    la2km=kms2deg[0]
    lo2km=kms2deg[1]
    
    i=np.argmin(ts)
    #i = 4
    t0=ts[i]-np.sqrt(((x0*lo2km-lo[i]*lo2km)**2)+((y0*la2km-la[i]*la2km)**2)+(el[i]-z0)**2)/vpin[i]  # initial guess origin time
    
    ni=0
    sols=[[t0,x0,y0,z0]]
    ndata = len(ts) # Number of data
    
    while 1:
        ni=ni+1
        D0=np.zeros(ndata)
        for i in range(ndata):
            D0[i] = np.sqrt(((lo[i]-x0)*lo2km)**2+((la[i]-y0)*la2km)**2+(el[i]-z0)**2)
        G=[]
        res=[]
        for i in range(ndata):
            vp = vpin[i]
            if(solvedep):
                G.append([1,((x0-lo[i])*lo2km)/(D0[i]*vp),((y0-la[i])*la2km)/(D0[i]*vp),(z0-el[i])/(D0[i]*vp)])
            else:
                G.append([1,((x0-lo[i])*lo2km)/(D0[i]*vp),((y0-la[i])*la2km)/(D0[i]*vp)])
            res.append(ts[i]-(D0[i]/vp+t0))
        G=np.array(G)
        res=np.array(res)
        #print(' ni ',ni)
        #print('G :\n',G[ni-1])
        #print('d :\n',d[ni-1])
        m=np.linalg.lstsq(G,res,rcond=None)[0]
        t0=t0+m[0]
        x0=x0+m[1]/lo2km # update longitude solution and convert to degrees
        y0=y0+m[2]/la2km # update latitude solution and convert to degrees
        if(solvedep): 
            z0=z0+m[3]
            dtol = np.sqrt((m[1]**2+m[2]**2+m[3]**2)) # distance moved by hypocentre
        else:
            dtol = np.sqrt(m[1]**2+m[2]**2)
        chisq = np.dot(res.T,res)
        if(verbose): print('Iteration :',ni,'Chi-sq:',chisq,' Change in origin time',m[0],' change in spatial distance:',dtol)
        sols.append([t0,x0,y0,z0])        
        if m[0]<tol[0] and dtol<tol[1]:
            break
        if(ni==nimax):
            print(' Maximum number of iterations reached in eqlocate. No convergence')
            break
    sols=np.array(sols)
    return sols, res

def plot_eq_solutions(MCsols,sols,title="Monte Carlo solutions",plotdepth=False):
    fig = plt.figure(figsize=(9,6))
    fig.suptitle(title, fontsize=20)

    ax1 = plt.subplot(221)
    ax1.plot(MCsols[:,1],MCsols[:,2], 'k+',lw=0.5)
    ax1.plot(sols[-1,1],sols[-1,2], 'ro')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    ax2 = plt.subplot(222)
    ax2.plot(MCsols[:,1],MCsols[:,0], 'k+',lw=0.5)
    ax2.plot(sols[-1,1],sols[-1,0], 'ro')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Origin Time')

    ax3 = plt.subplot(223)
    ax3.plot(MCsols[:,2],MCsols[:,0], 'k+',lw=0.5)
    ax3.plot(sols[-1,2],sols[-1,0], 'ro')
    ax3.set_xlabel('Latitude')
    ax3.set_ylabel('Origin Time')

    if(plotdepth):
        ax4 = plt.subplot(224)
        ax4.plot(MCsols[:,3],MCsols[:,0], 'k+',lw=0.5)
        ax4.plot(sols[-1,3],sols[-1,0], 'ro')
        ax4.set_xlabel('Depth')
        ax4.set_ylabel('Origin Time')

    plt.show()

# ----------------------------------------------------------------------------
# Calculate covariance matrix from error distribution for each pair of solution parameters
# ----------------------------------------------------------------------------

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
           
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_eq_conf_ellipses(sols,Cm_cov,title="Error ellipses for solution"):
    l68 = np.sqrt(stats.chi2.ppf(q=0.68,df=2)) # Number of standard deviations equivalent to 68% confidence ellipse
    l95 = np.sqrt(stats.chi2.ppf(q=0.95,df=2)) # Number of standard deviations equivalent to 95% confidence ellipse
	
    CmProj01 = Cm_cov[np.ix_([1,0],[1,0])]
    CmProj02 = Cm_cov[np.ix_([2,0],[2,0])]
    CmProj12 = Cm_cov[np.ix_([1,2],[1,2])]
    s0,s1,s2,st = np.sqrt(np.diag(Cm_cov))

    fig = plt.figure(figsize=(9,6))
    fig.suptitle(title, fontsize=16)

    ax1 = plt.subplot(221)
    plot_cov_ellipse(CmProj12,sols[-1][1:], ax=ax1,nstd=l68,color='Blue',alpha=0.4,label='68% Confidence')
    plot_cov_ellipse(CmProj12,sols[-1][1:], ax=ax1,nstd=l95,color='Green',alpha=0.4,label='95% Confidence')
    ax1.set_xlim(sols[-1][1]-1.3*1.96*s1,sols[-1][1]+1.3*1.96*s1)
    ax1.set_ylim(sols[-1][2]-1.3*1.96*s2,sols[-1][2]+1.3*1.96*s2)
    ax1.set_xlabel('Longitude (deg)')
    ax1.set_ylabel('Latitude (deg)')

    ax2 = plt.subplot(222)
    plot_cov_ellipse(CmProj01,[sols[-1][1],sols[-1][0]], ax=ax2,nstd=l68,color='Blue',alpha=0.4,label='68% Confidence')
    plot_cov_ellipse(CmProj01,[sols[-1][1],sols[-1][0]], ax=ax2,nstd=l95,color='Green',alpha=0.4,label='95% Confidence')
    ax2.set_xlim(sols[-1][1]-1.3*1.96*s1,sols[-1][1]+1.3*1.96*s1)
    ax2.set_ylim(sols[-1][0]-1.3*1.96*s0,sols[-1][0]+1.3*1.96*s0)
    ax2.set_xlabel('Longitude (deg)')
    ax2.set_ylabel('Origin Time (s)')
	
    ax3 = plt.subplot(223)
    plot_cov_ellipse(CmProj02,[sols[-1][2],sols[-1][0]], ax=ax3,nstd=l68,color='Blue',alpha=0.4,label='68% Confidence')
    plot_cov_ellipse(CmProj02,[sols[-1][2],sols[-1][0]], ax=ax3,nstd=l95,color='Green',alpha=0.4,label='95% Confidence')
    ax3.set_xlim(sols[-1][2]-1.3*1.96*s2,sols[-1][2]+1.3*1.96*s2)
    ax3.set_ylim(sols[-1][0]-1.3*1.96*s0,sols[-1][0]+1.3*1.96*s0)
    ax3.set_xlabel('Latitude (deg)')
    ax3.set_ylabel('Origin Time (s)')
    plt.show()
