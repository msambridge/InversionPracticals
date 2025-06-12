def plotRFmod(RFo,time1,model,RF2=None): # Receiver function and model plotting utility
    '''
    Plot RFmod
    
    Inputs:
        RF, numpy.ndarray(n,)        : Amplitude of receiver function
        time, numpy.ndarray(n,)      : Time points of receiver function
        model, numpy.ndarray(L,3)    : Velocity model as triplet of (depth,S-wavespeed,VpVs ratio) for L layers
    
    If RF2 is not None then it is assumed RF2=[RFp,time2] and a second RF is plotted.
    '''    
    px = np.zeros([2*len(model),2])
    py = np.zeros([2*len(model),2])
    n=len(model)
    px[0::2,0],px[1::2,0],px[1::2,1],px[2::2,1] = model[:,1],model[:,1],model[:,0],model[:-1,0]


    f, (a0, a1) = plt.subplots(1,2, figsize=(12,4), gridspec_kw = {'width_ratios':[1, 3]})

    a1.set_title('Velocity model and receiver functions')
    a1.set_xlabel("Time (s)")
    a1.set_ylabel("Amplitude")
    a1.grid(True)
    a1.plot(time1, RFo, 'r-',label='Observed')
    if(RF2 is not None):a1.plot(RF2[1], RF2[0], label='Predicted')
    a1.legend()

    a0.set_title(" Velocity model")                   # Plot velocity model with Receiver function
    a0.set_xlabel('Vs (km/s)')
    a0.set_ylabel('Depth (km)')
    a0.plot(px[:,0],px[:,1],'b-')
    a0.invert_yaxis()

    #plt.tight_layout()
    plt.show()
