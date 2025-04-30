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
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from anu_inversion_course import plotcovellipse as pc
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
                        pc.plot_cov_ellipse(CProj,[m[i],m[j]], ax=ax,nstd=cl,alpha=0.4,label=str(confint[ii])+"% Confidence")
                        ax.plot(m[i],m[j],'k.')
                    
                    if(isinstance(C2,np.ndarray) & isinstance(m2,np.ndarray)):
                        sigx2 = np.sqrt(C2[i,i])
                        sigy2 = np.sqrt(C2[j,j])
                        CProj2 = C2[np.ix_([i,j],[i,j])]
                        for ii,cl in enumerate(c):
                            pc.plot_cov_ellipse(CProj2,[m2[i],m2[j]], ax=ax,nstd=cl,alpha=0.4,label=str(confint[ii])+"% Confidence",fill=False)
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
