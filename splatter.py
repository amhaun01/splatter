import numpy as np
from numpy import fft
from numpy.random import randn, rand
from matplotlib import cm
from scipy import ndimage

def getSplat(impar,noisin,cmapname='summer'):
    np.random.seed(impar['rseed'])

    S = impar['S']
    #sf grid
    xs = np.array(range(0,S))-S/2
    ys = np.array(range(0,S))-S/2
    xx,yy = np.meshgrid(xs,ys)
    xxyy = xx + 1j*yy

    #spatial frequencies
    fs = fft.fftshift(np.abs(xxyy))

    #the coarse-scale window, image-size
    #always the same for all images, so precompute is fine
    sig2 = 1
    filt2 = np.exp(-.5*(fs/sig2)**2)

    #input colormap
    cmap = cm.get_cmap(cmapname)
    corder = np.random.permutation(256)
    palette = []
    #color palette
    for x in range(0,impar['NCOL']):
        palette.append(cmap(corder[x]/255))


    #background brightness
    bgr = rand()
    if bgr<(1/3):
        bgb = 0.15
        bgc = 'dark'
    elif bgr<2/3:
        bgb = 0.55
        bgc = 'gray'
    else:
        bgb = 0.95
        bgc = 'light'

    #CFLIP: flip direction (left/right or top/bottom) - will be coherent throughout the image
    CFLIP = [randn()>0, randn()>0]
    #DRIFT: use correlated sig1 across iterations
    DRIFT = randn()>0

    img = bgb*np.ones(shape=(S,S,3)) - .1*rand(S,S,3)

    for C in range(0,impar['NCOL']):
        #amplitude modulation noise
        nsamp2 = rand(S,S)
        wind = np.real(fft.ifft2(filt2*fft.fft2(nsamp2)))

        #the iterated layer
        fv = np.ones(shape=(S,S))<0

        if DRIFT:
            sig1 = (S/32 -1)*rand()
        for R in range(0,impar['RS']):
            if DRIFT:
                sig1 = max(min(sig1*(1 + .2*randn()),(S/8 - 1)),1)
            else:
                sig1 = 1 + 2**np.log2((S/8-1)*rand()*rand())

            #print(sig1)
            #filter and normalize the input signal
            filt1 = np.exp(-.5*(fs/sig1)**2)
            fn = np.real(fft.ifft2(filt1*fft.fft2(noisin)))
            fn2 = wind*(fn - np.min(fn))/(np.max(fn)-np.min(fn))
            #level to cut the filtered input
            lim = np.max(fn2)*rand()
            #thickness of the cut
            lim_thresh = .01 + .05*rand()

            vv = np.array(fn2>lim) & np.array(fn2<(lim+lim_thresh))
            #this is the core operation: concatenating the slices over iterations R, within each color C
            fv = fv | vv
            #fv = np.logical_or(fv,np.logical_and(fn2>lim,fn2<(lim+lim_thresh)))

        #we'll use fb as a smooth local intensity weight, and for the clipping
        sig3 = (S/64)*rand()+1
        filt3 = np.exp(-.5*(fs/sig3)**2)
        fb = np.real(fft.ifft2(filt3*fft.fft2(fv)))
        fb = .5+.5*(fb - np.min(fb))/(np.max(fb)-np.min(fb))

        if randn()>0:#whether or not to clip a layer according to the local derivative
            if randn()>0:#clipping L/R (true) or U/D (false)
                if CFLIP[0]:#pre-set clipping direction
                    fv = np.logical_and(fv,fb>np.roll(fb,1,axis=1))
                else:
                    fv = np.logical_and(fv,fb<np.roll(fb,1,axis=1))
            else:
                if CFLIP[1]:
                    fv = np.logical_and(fv,fb>np.roll(fb,1,axis=0))
                else:
                    fv = np.logical_and(fv,fb<np.roll(fb,1,axis=0))

        #apply the intensity weight
        fvn = fv*fb
        #edges to add definition to each layer
        sx = ndimage.sobel(fv.astype(int),axis=0,mode='wrap')
        sy = ndimage.sobel(fv.astype(int),axis=1,mode='wrap')
        evT = np.hypot(sx,sy)
        ev = evT>(.8*np.max(evT))#need a threshold, so....
        #ev = ndimage.sobel(fv.astype(int))

        for X in [0,1,2]:
            img[:,:,X] = (1-fv)*img[:,:,X] + palette[C][X]*fvn
            #img[:,:,X] = img[:,:,X] - .5*(1-palette[C][X])*ev.astype(float)
            img[:,:,X] = img[:,:,X]*(1-.2*ev)
    #print('edge drawing is disabled')
    return img, ev
