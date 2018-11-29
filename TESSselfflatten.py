import numpy as np
from astropy.io import fits
import sys
import pylab as p
p.ion()
import warnings
warnings.filterwarnings('once')


def dopolyfit(win,d,ni,sigclip):
    base = np.polyfit(win[:,0],win[:,1],w=1.0/win[:,2],deg=d)
    #for n iterations, clip sigma, redo polyfit
    for iter in range(ni):
        #winsigma = np.std(win[:,1]-np.polyval(base,win[:,0]))
        offset = np.abs(win[:,1]-np.polyval(base,win[:,0]))/win[:,2]
        
        if (offset<sigclip).sum()>int(0.8*len(win[:,0])):
            clippedregion = win[offset<sigclip,:]
        else:
            clippedregion = win[offset<np.average(offset)]
            
        base = np.polyfit(clippedregion[:,0],clippedregion[:,1],w=1.0/np.power(clippedregion[:,2],2),deg=d)
    return base

def CheckForGaps(dat,centidx,winlowbound,winhighbound,gapthresh):
    diffshigh = np.diff(dat[centidx:winhighbound,0])
    gaplocshigh = np.where(diffshigh>gapthresh)[0]
    highgap = len(gaplocshigh)>0
    diffslow = np.diff(dat[winlowbound:centidx,0])
    gaplocslow = np.where(diffslow>gapthresh)[0]
    lowgap = len(gaplocslow)>0
    return lowgap, highgap, gaplocslow,gaplocshigh

def formwindow(datcut,dat,cent,size,boxsize,gapthresh,expectedpoints,cadence):
    winlowbound = np.searchsorted(datcut[:,0],cent-size/2.)
    winhighbound = np.searchsorted(datcut[:,0],cent+size/2.)
    boxlowbound = np.searchsorted(dat[:,0],cent-boxsize/2.)
    boxhighbound = np.searchsorted(dat[:,0],cent+boxsize/2.)
    centidx = np.searchsorted(datcut[:,0],cent)

    if centidx==boxlowbound:
        centidx += 1
    if winhighbound == len(datcut[:,0]):
        winhighbound -= 1
    flag = 0

    lowgap, highgap, gaplocslow,gaplocshigh = CheckForGaps(datcut,centidx,winlowbound,winhighbound,gapthresh)

    if winlowbound == 0:
        lowgap = True
        gaplocslow = [-1]
    if winhighbound == len(datcut[:,0]):
         highgap = True
         gaplocshigh = [len(datcut[:,0]) -centidx]
    
    if highgap:
        if lowgap:
            winhighbound = centidx + gaplocshigh[0]
            winlowbound = winlowbound + 1 + gaplocslow[-1]
        else:
            winhighbound = centidx + gaplocshigh[0]
            winlowbound = np.searchsorted(datcut[:,0],datcut[winhighbound,0]-size)
            lowgap, highgap, gaplocslow,gaplocshigh = CheckForGaps(datcut,centidx,winlowbound,winhighbound,gapthresh)
            if lowgap:
                winlowbound = winlowbound + 1 + gaplocslow[-1] #uses reduced fitting section
    else:
        if lowgap:
            winlowbound = winlowbound + 1 + gaplocslow[-1]            
            winhighbound =  np.searchsorted(datcut[:,0],datcut[winlowbound,0]+size)
            lowgap, highgap, gaplocslow,gaplocshigh = CheckForGaps(datcut,centidx,winlowbound,winhighbound,gapthresh)
            if highgap:
                winhighbound = centidx + gaplocshigh[0] #uses reduced fitting section

    #window = np.concatenate((dat[winlowbound:boxlowbound,:],dat[boxhighbound:winhighbound,:]))
    window = datcut[winlowbound:winhighbound,:]
    if len(window[:,0]) < 20:
        flag = 1
    box = dat[boxlowbound:boxhighbound,:]

    return window,boxlowbound,boxhighbound,flag

def polyflatten(lc,winsize,stepsize,polydegree,niter,sigmaclip,gapthreshold,t0=0.,plot=False,transitcut=False,tc_per=0,tc_t0=0,tc_tdur=0,outfile=False):

    lcdetrend = np.zeros(len(lc[:,0]))

    #general setup
    lenlc = lc[-1,0]
    nsteps = np.ceil(lenlc/stepsize).astype('int')
    stepcentres = np.arange(nsteps)/float(nsteps) * lenlc + stepsize/2.
    cadence = np.median(np.diff(lc[:,0]))
    
    expectedpoints = winsize/2./cadence

    if transitcut:
        if transitcut > 1:
            timecut, fluxcut = lc[:,0].copy()+t0, lc[:,1].copy()
            errcut, qualitycut = lc[:,2].copy(), lc[:,3].copy()
            for cutidx in range(transitcut):
                timecut, fluxcut, errcut = CutTransits(timecut,fluxcut,errcut,
                										tc_t0[cutidx],
                										tc_per[cutidx],tc_tdur[cutidx])
        else:
            timecut, fluxcut, errcut = CutTransits(lc[:,0]+t0,lc[:,1],
            										lc[:,2],tc_t0,tc_per,tc_tdur)
        lc_tofit = np.zeros([len(timecut),3])
        lc_tofit[:,0] = timecut-t0
        lc_tofit[:,1] = fluxcut
        lc_tofit[:,2] = errcut
    else:
        lc_tofit = lc


    #for each step centre:
    for s in range(nsteps):
        stepcent = stepcentres[s]
        winregion,boxlowbound,boxhighbound,flag = formwindow(lc_tofit,lc,stepcent,winsize,stepsize,gapthreshold,expectedpoints,cadence)  #should return window around box not including box

        if not flag:
            baseline = dopolyfit(winregion,polydegree,niter,sigmaclip)
            lcdetrend[boxlowbound:boxhighbound] = lc[boxlowbound:boxhighbound,1] / np.polyval(baseline,lc[boxlowbound:boxhighbound,0])
        else:
            lcdetrend[boxlowbound:boxhighbound] = np.ones(boxhighbound-boxlowbound)
    
    output = np.zeros_like(lc)
    output[:,0] = lc[:,0] + t0
    output[:,1] = lcdetrend
    output[:,2] = lc[:,2]
    
    if plot:
        p.figure(1)
        p.clf()
        p.plot(lc[:,0]+t0,lc[:,1],'b.')
        p.plot(output[:,0],output[:,1],'g.')
        input('Press any key to continue')
    if outfile:
        np.savetxt(outfile,output)
    else:
        return output

def CutTransits(time,flux,err,transitt0,transitper,transitdur):
    phase = np.mod(time-transitt0,transitper)/transitper    
    intransit = (phase<transitdur/transitper) | (phase>1-transitdur/transitper)
    return time[~intransit],flux[~intransit],err[~intransit]

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def TESSflatten(lcurve, kind='butter', split=True, highcut=12., winsize=3.5, 
				stepsize=0.15,polydeg=3,niter=10,sigmaclip=4.,gapthresh=100.):
    #kind in butter or poly
    cadence = np.median(np.diff(lcurve[:,0]))
    flatlc = np.zeros(1)
    
    if split:
        #treat each orbit separately
        norbits = np.round((lcurve[-1,0]-lcurve[0,0]) / 13.94).astype('int')
        for orbit in range(norbits):
            start = np.searchsorted(lcurve[:,0],orbit*13.94)
            end = np.searchsorted(lcurve[:,0],(orbit+1)*13.94)
            lcseg = lcurve[start:end,:]
            
            if kind=='butter':
                from scipy import signal
                fs = 1./(cadence*86400.) #sample rate (Hz)
                highcutHz = 1./(highcut*60*60) #cut off frequency (Hz)
                lcseg_flat = butter_highpass_filter(lcseg[:,1], highcut, fs, order=6)
            elif kind=='poly':
                lcseg_flat = polyflatten(lcseg,winsize,stepsize,polydeg,niter,
                							sigmaclip,gapthresh)[:,1]
             
            flatlc = np.hstack((flatlc,lcseg_flat))
    flatlc = np.array(flatlc)
    return flatlc[1:]         

