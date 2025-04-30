import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

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

# ----------------------------------------------------------------------------
# Calculate covariance matrix from error distribution for each pair of solution parameters
# ----------------------------------------------------------------------------

def plot_conf_ellipses(mls,Cm,title="Error ellipses for solution",limits=None,CIs=None):
    CmProj01 = Cm[np.ix_([0,1],[0,1])]
    CmProj02 = Cm[np.ix_([0,2],[0,2])]
    CmProj12 = Cm[np.ix_([1,2],[1,2])]

    sig_param1 = np.sqrt(Cm[0,0]) # standard error for parameter 1
    sig_param2 = np.sqrt(Cm[1,1]) # standard error for parameter 2
    sig_param3 = np.sqrt(Cm[2,2]) # standard error for parameter 3

    l68 = np.sqrt(stats.chi2.ppf(q=0.68,df=2)) # number of standard deviations equivalent to 68% confidence ellipse
    l95 = np.sqrt(stats.chi2.ppf(q=0.95,df=2)) # number of standard deviations equivalent to 95% confidence ellipse

    fig = plt.figure(figsize=(9,6))
    fig.suptitle(title, fontsize=16)

    ax1 = plt.subplot(221)

    if(limits==None):
       ax1.set_xlim(mls[0]-1.3*1.96*sig_param1,mls[0]+1.3*1.96*sig_param1)
       ax1.set_ylim(mls[1]-1.3*1.96*sig_param2,mls[1]+1.3*1.96*sig_param2)
    else:
       ax1.set_xlim(limits[0][0][0],limits[0][0][1])
       ax1.set_ylim(limits[0][1][0],limits[0][1][1])

    ax1.set_xlabel('m1')
    ax1.set_ylabel('m2')

    ax2 = plt.subplot(222)
    if(limits==None):
        ax2.set_xlim(mls[1]-1.3*1.96*sig_param2,mls[1]+1.3*1.96*sig_param2)
        ax2.set_ylim(mls[2]-1.3*1.96*sig_param3,mls[2]+1.3*1.96*sig_param3)
    else:
       ax2.set_xlim(limits[1][0][0],limits[1][0][1])
       ax2.set_ylim(limits[1][1][0],limits[1][1][1])

    ax2.set_xlabel('m2')
    ax2.set_ylabel('m3')

    ax3 = plt.subplot(223)
    if(limits==None):
        ax3.set_xlim(mls[0]-1.3*1.96*sig_param1,mls[0]+1.3*1.96*sig_param1)
        ax3.set_ylim(mls[2]-1.3*1.96*sig_param3,mls[2]+1.3*1.96*sig_param3)
    else:
       ax3.set_xlim(limits[2][0][0],limits[2][0][1])
       ax3.set_ylim(limits[2][1][0],limits[2][1][1])

    ax3.set_xlabel('m1')
    ax3.set_ylabel('m3')

    if(CIs==None):
       plot_cov_ellipse(CmProj01,mls[0:2], ax=ax1,nstd=l68,color='Blue',alpha=0.4,label="68% Confidence")
       plot_cov_ellipse(CmProj01,mls[0:2], ax=ax1,nstd=l95,color='Green',alpha=0.4,label="95% Confidence")
       plot_cov_ellipse(CmProj12,mls[1:], ax=ax2,nstd=l68,color='Blue',alpha=0.4,label='68% Confidence')
       plot_cov_ellipse(CmProj12,mls[1:], ax=ax2,nstd=l95,color='Green',alpha=0.4,label='95% Confidence')
       plot_cov_ellipse(CmProj02,[mls[0],mls[2]], ax=ax3,nstd=l68,color='Blue',alpha=0.4,label='68% Confidence')
       plot_cov_ellipse(CmProj02,[mls[0],mls[2]], ax=ax3,nstd=l95,color='Green',alpha=0.4,label='95% Confidence')
    else:
       colors = ['Blue','Green','Yellow']
       for j,l, in enumerate(CIs):
           plot_cov_ellipse(CmProj01,mls[0:2], ax=ax1,nstd=l,color=colors[j],alpha=0.4)
           plot_cov_ellipse(CmProj12,mls[1:], ax=ax2,nstd=l,color=colors[j],alpha=0.4)
           plot_cov_ellipse(CmProj02,[mls[0],mls[2]], ax=ax3,nstd=l,color=colors[j],alpha=0.4)

    ax1.plot(mls[0],mls[1],'ro')
    ax2.plot(mls[1],mls[2],'ro')
    ax3.plot(mls[0],mls[2],'ro')

    plt.show()

##########################
# Random draws of feasible solutions
##########################

def plot_feasible(mls,Cm,points,limits=None):
    fig = plt.figure(figsize=(9,6))
    fig.suptitle("Random draws of feasible solutions", fontsize=20)

    sig_param1 = np.sqrt(Cm[0,0]) # standard error for parameter 1
    sig_param2 = np.sqrt(Cm[1,1]) # standard error for parameter 2
    sig_param3 = np.sqrt(Cm[2,2]) # standard error for parameter 3

    #points = np.random.multivariate_normal(mean=mls, cov=Cm, size=size)

    ax1 = plt.subplot(221)
    xp, yp = points.T[0], points.T[1]
    ax1.plot(xp, yp, 'k+')
    ax1.plot(mls[0],mls[1], 'ro')
    if(limits==None):
       ax1.set_xlim(mls[0]-1.3*1.96*sig_param1,mls[0]+1.3*1.96*sig_param1)
       ax1.set_ylim(mls[1]-1.3*1.96*sig_param2,mls[1]+1.3*1.96*sig_param2)
    else:
       ax1.set_xlim(limits[0][0][0],limits[0][0][1])
       ax1.set_ylim(limits[0][1][0],limits[0][1][1])

    ax1.set_xlabel('m1')
    ax1.set_ylabel('m2')

    ax2 = plt.subplot(222)
    xp, yp =  points.T[1], points.T[2]
    ax2.plot(xp, yp, 'k+')
    ax2.plot(mls[1],mls[2], 'ro')
    if(limits==None):
        ax2.set_xlim(mls[1]-1.3*1.96*sig_param2,mls[1]+1.3*1.96*sig_param2)
        ax2.set_ylim(mls[2]-1.3*1.96*sig_param3,mls[2]+1.3*1.96*sig_param3)
    else:
       ax2.set_xlim(limits[1][0][0],limits[1][0][1])
       ax2.set_ylim(limits[1][1][0],limits[1][1][1])
    ax2.set_xlabel('m2')
    ax2.set_ylabel('m3')

    ax3 = plt.subplot(223)
    xp, yp =  points.T[0], points.T[2]
    ax3.plot(xp, yp, 'k+')
    ax3.plot(mls[0],mls[2], 'ro')
    if(limits==None):
        ax3.set_xlim(mls[0]-1.3*1.96*sig_param1,mls[0]+1.3*1.96*sig_param1)
        ax3.set_ylim(mls[2]-1.3*1.96*sig_param3,mls[2]+1.3*1.96*sig_param3)
    else:
       ax3.set_xlim(limits[2][0][0],limits[2][0][1])
       ax3.set_ylim(limits[2][1][0],limits[2][1][1])
    ax3.set_xlabel('m1')
    ax3.set_ylabel('m3')

    plt.show()

##########################
# Random draws of feasible solutions
##########################

def plot_sols_scatter(mls,points,title=None,limits=None):
    fig = plt.figure(figsize=(9,6))
    if(title is not None): fig.suptitle(title,fontsize=20)

    ax1 = plt.subplot(221)
    xp, yp = points.T[0], points.T[1]
    ax1.plot(xp, yp, 'k+')
    ax1.plot(mls[0],mls[1], 'ro')

    ax1.set_xlabel('m1')
    ax1.set_ylabel('m2')

    ax2 = plt.subplot(222)
    xp, yp =  points.T[1], points.T[2]
    ax2.plot(xp, yp, 'k+')
    ax2.plot(mls[1],mls[2], 'ro')

    ax2.set_xlabel('m2')
    ax2.set_ylabel('m3')

    ax3 = plt.subplot(223)
    xp, yp =  points.T[0], points.T[2]
    ax3.plot(xp, yp, 'k+')
    ax3.plot(mls[0],mls[2], 'ro')
    ax3.set_xlabel('m1')
    ax3.set_ylabel('m3')

    plt.show()

def plot_marginal_histograms(mls,MCsols):

    Cm_std= np.std(MCsols,axis=0)
    sig_param1 = Cm_std[0]
    sig_param2 = Cm_std[1]
    sig_param3 = Cm_std[2]
    n = len(MCsols)

    fig = plt.figure()
    fig.suptitle("Monte Carlo solutions histogram", fontsize=20)
    n, bins, patches = plt.hist(MCsols.T[0], 50, facecolor='g', alpha=0.75)
    plt.plot([mls[0]-1.96*sig_param1,mls[0]-1.96*sig_param1],[0.0,0.85*np.max(n)],'r:')
    plt.plot([mls[0]+1.96*sig_param1,mls[0]+1.96*sig_param1],[0.0,0.85*np.max(n)],'r:')
    plt.xlabel('m1')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()

    fig = plt.figure()
    fig.suptitle("Monte Carlo solutions histogram", fontsize=20)
    n, bins, patches = plt.hist(MCsols.T[1], 50, facecolor='g', alpha=0.75)
    plt.plot([mls[1]-1.96*sig_param2,mls[1]-1.96*sig_param2],[0.0,0.85*np.max(n)],'r:')
    plt.plot([mls[1]+1.96*sig_param2,mls[1]+1.96*sig_param2],[0.0,0.85*np.max(n)],'r:')
    plt.xlabel('m2')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()

    fig = plt.figure()
    fig.suptitle("Monte Carlo solutions histogram", fontsize=20)
    n, bins, patches = plt.hist(MCsols.T[2], 50, facecolor='g', alpha=0.75)
    plt.plot([mls[2]-1.96*sig_param3,mls[2]-1.96*sig_param3],[0.0,0.85*np.max(n)],'r:')
    plt.plot([mls[2]+1.96*sig_param3,mls[2]+1.96*sig_param3],[0.0,0.85*np.max(n)],'r:')
    plt.xlabel('m3')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()

def gaussian_corner(C,m,confint=[68,95],C2=None,m2=None,labels=None,title=None,figsizex=None,figsizey=None,filename=None,checkinput=True):
    '''
    
        Plots 2D projections of a multi-variate Gaussian distribution in a `corner` style plot.

            Inputs:
                C - ndarray, shape(ndim,ndim): Covariance matrix of multi-dimensional hyper-ellipse with dimension ndim.
                m - ndarray, shape(ndim): mean vector of multi-dimensional hyper-ellipse.
                confint - list, integers or floats: percentages of confidence ellipse to be plotted (defaults 68%, 95%).
                C2 - ndarray, shape(ndim,ndim): covariance matrix of second multi-dimensional hyper-ellipse to plotted with first (Optional)
                m2 - ndarray, shape(ndim): mean vector of second multi-dimensional hyper-ellipse (Optional).
                labels - list of strings: lables for each parameter axis.
                title - string as title for plot.
                figsizex - float, size of plot in x-direction passed to matplotlib.figure()
                figsizey - float, size of plot in y-direction passed to matplotlib.figure()
                filename - string: if set, string passed to matplotlib.savefig()
                checkinput - bool: if True (default) check eigenvalues of input covraiance matrix for positive definiteness.
                
            Creates a corner style plot for each pair of parameters in C, with projections of the confidence contours plotted corresponding 
            to the multi-dimnensional Gaussian Probability density function (x-m)^T C (x-m). If C2, m2 are included these are plotted as 
            contours only without a colour fill. If C2, m2 are included the axes limits are scaled to include all confidence ellipses. 
            
                
    '''
    ndim = np.shape(C)[0]
    if(checkinput): # check whether input matrix is positive definite
        w,v = np.linalg.eig(C)
        if(np.min(w)<=0.0):
            print(' Error: Input matrix is not positive definite')
            print(' Eigenvalues of input matrix :\n',w)
            return
        
    if(figsizex == None): figsizex = 9
    if(figsizey == None): figsizey = 9
    if(title == None): title = 'Gauss_corner plot'
    if(labels==None): 
        labels = []
        for i in range(ndim):
            labels += ["m"+str(i+1)]
    fig = plt.figure(figsize=(figsizex,figsizey))
    plt.suptitle(title)

    c,c1d = np.zeros(len(confint)),np.zeros(len(confint))
    for i,p in enumerate(confint):
        c[i] = np.sqrt(stats.chi2.ppf(q=p/100.,df=2))
        #print(' Number of standard deviations for '+str(p)+'% Conf ellipse = ',c[i])
        c1d[i] = np.sqrt(stats.chi2.ppf(q=p/100.,df=1))
    fac = 1.3*np.max(c1d)
    
    for i in range(ndim):
        sigx = np.sqrt(C[i,i])
        x0,x1 = m[i]-fac*sigx,m[i]+fac*sigx
        for j in range(ndim):
            sigy = np.sqrt(C[j,j])
            y0,y1 = m[j]-fac*sigy,m[j]+fac*sigy
            if(i<=j):
                k = 1+j*ndim+i
                ax = plt.subplot(ndim,ndim,k)
                if(i!=j):
                    CProj = C[np.ix_([i,j],[i,j])]
                    for ii,cl in enumerate(c):
                        plot_cov_ellipse(CProj,[m[i],m[j]], ax=ax,nstd=cl,alpha=0.4,label=str(confint[ii])+"% Confidence")
                        ax.plot(m[i],m[j],'k.')
                    
                    if(isinstance(C2,np.ndarray) & isinstance(m2,np.ndarray)):
                        sigx2 = np.sqrt(C2[i,i])
                        sigy2 = np.sqrt(C2[j,j])
                        CProj2 = C2[np.ix_([i,j],[i,j])]
                        for ii,cl in enumerate(c):
                            plot_cov_ellipse(CProj2,[m2[i],m2[j]], ax=ax,nstd=cl,alpha=0.4,label=str(confint[ii])+"% Confidence",fill=False)
                            ax.plot(m2[i],m2[j],'k.')
                        x02,x12 = m2[i]-fac*sigx2,m2[i]+fac*sigx2
                        y02,y12 = m2[j]-fac*sigy2,m2[j]+fac*sigy2
                        x0 = np.min([x0,x02])
                        y0 = np.min([y0,y02])
                        x1 = np.max([x1,x12])
                        y1 = np.max([y1,y12])
                    
                    if(j==ndim-1 ):ax.set_xlabel(labels[i])
                    if(i==0):ax.set_ylabel(labels[j])
                    if(j!=ndim-1): ax.axes.xaxis.set_ticklabels([])
                    if(i!=0): ax.axes.yaxis.set_ticklabels([])
                    ax.set_xlim(x0,x1)
                    ax.set_ylim(y0,y1)
                    
                else:
                    #pass
                    if(j==ndim-1 ):
                        ax.set_xlabel(labels[i])
                    else:
                        ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    
                    x = np.linspace(m[i]-fac*sigx, m[i]+fac*sigx, 100)
                    y = stats.norm.pdf(x, m[i], sigx)
                    ax.plot(x,y)
                    for ii,cl in enumerate(c1d):
                        line, = ax.plot([m[i]-c1d[ii]*sigx,m[i]-c1d[ii]*sigx],[0.0,0.95*np.max(y)],':')
                        ax.plot([m[i]+c1d[ii]*sigx,m[i]+c1d[ii]*sigx],[0.0,0.95*np.max(y)],':',color = line.get_color())
                    if(isinstance(C2,np.ndarray) & isinstance(m2,np.ndarray)):
                        sigx2 = np.sqrt(C2[i,i])
                        x02,x12 = m2[i]-fac*sigx2,m2[i]+fac*sigx2
                        x0 = np.min([x0,x02])
                        x1 = np.max([x1,x12])
                    ax.set_xlim(x0,x1)
                
                    
    if(filename != None): plt.savefig(filename)
    plt.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    #-- Example usage -----------------------
    # Generate some random, correlated data
    points = np.random.multivariate_normal(
            mean=(1,1), cov=[[1.0, .2],[0.2, 3.0]], size=1000)
    # Plot the raw points...
    x, y = points.T
    plt.plot(x, y, 'ro')

    # Plot a transparent 3 standard deviation covariance ellipse
    plot_point_cov(points, nstd=3, alpha=0.5, color='green')

    plt.show()
