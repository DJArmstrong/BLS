# -*- coding: utf-8 -*-
import numpy as np
import csv
import pylab as p
p.ion()
import sys
import os
import matplotlib.cm as cm
from astropy.io import fits
import random


def Run(N, jump=0):

    ### Set the number of eyeballers here. CRUCIAL FOR MAKING SURE EVERY LIGHTCURVE IS SEEN ###
    n_eyeballs = 1.
    TICinfofile = ''
    LCdir = '/Users/davidarmstrong/Data/TESS/NCORES/TOI-220/'  #should be full of files called 'TICID_lc.txt' or '.fits'. Format can be changed in Eyeball
    BLSfile = '/Users/davidarmstrong/Data/TESS/NCORES/TOI-220/TESS_detns_full.csv'
    ######

    N=int(N)
    fnamein = input('ENTER UNIQUE NAME (existing or new):')    
    
    newfname = 'TESS_S01_'+fnamein+'_'+str(N)+'.csv'
    
    BLSdets = np.genfromtxt(BLSfile,delimiter=',',names=True)
    
    if os.path.exists(TICinfofile):
        TICinfo = np.genfromtxt(TICinfofile,delimiter=',',names=True)
    else:
        TICinfo = None
        
    #consider also saving flattened lcs rather than reflattening to plot. currently just doesn't flatten
    if not os.path.exists(BLSfile[:-4]+'_TIClist.txt'):
        uniqueTICS, unique_idx = np.unique(BLSdets['TICID'],return_index=True)
        #sort by SNR
        uniqueTICS = uniqueTICS[np.argsort(BLSdets['SNR_1'][unique_idx])][::-1]
        np.savetxt(BLSfile[:-4]+'_TIClist.txt',uniqueTICS)
    else:
        uniqueTICS = np.genfromtxt(BLSfile[:-4]+'_TIClist.txt')
        uniqueTICS = np.atleast_1d(uniqueTICS)
    
    uniqueTICS = uniqueTICS.astype('int')
    
    if len(uniqueTICS)==1: #one TIC only
        TIC = uniqueTICS[0]
        pgramfile = BLSfile[:-4]+'_'+str(TIC)+'_0_pgram.txt'
        Eyeball(newfname, TIC, N, BLSdets[BLSdets['TICID']==TIC], LCdir, pgramfile, TICinfo)    
    else:
        n, go = 0, 0
        for TIC in uniqueTICS:
            if int(n/n_eyeballs)>=int(jump):
                go=1

            if n%int(n_eyeballs)==N and go==1:
                pgramfile = BLSfile[:-4]+'_'+str(TIC)+'_0_pgram.txt'
                print(str(int(n/n_eyeballs))+'/'+str(len(uniqueTICS)/n_eyeballs)+' Lightcurves analysed.')
                Eyeball(newfname, TIC, N, BLSdets[BLSdets['TICID']==TIC], LCdir, pgramfile, TICinfo)
            n+=1


def PlotLC(lc, offset=0,  col=0,  bin=0, fig=1, titl='Lightcurve', hoff=0, ephem=None):
    '''This function plots a lightcurve (lc) in format: times, flux, flux_errs.
    Offset allows multiple lightcurves to be displayed above each other. Col allows colour to be varied (otherwise, random)
    You return nothing, Jon Snow.'''
    if col==0:
        col='#28596C'
        #col= '#'+"".join([random.choice('0123456789abcdef') for n in xrange(6)])
    p.figure(fig, figsize=[16.0,6.]); p.title(titl);p.xlabel('Time');p.ylabel('Relative flux');
    p.errorbar(lc[:, 0]+hoff, lc[:, 1]+offset, yerr=lc[:, 2], fmt='.',color='#BBBBBB')
    p.plot(lc[:, 0]+hoff, lc[:, 1]+offset, '.',color=col)
      
    if bin!=0:
        binlc = BinLC(lc, int((lc[-1, 0]-lc[0, 0])/bin))
        #col='#663366'
        col = 'r'
        #col='#'+"".join([random.choice('0123456789') for n in xrange(6)]) if col==0 else '#666666'
        p.errorbar(binlc[:, 0], binlc[:, 1]+offset, yerr=binlc[:, 2], fmt='.',color=col)

    if ephem is not None:
        epoch = ephem[1]
        period = ephem [0]
        while epoch > lc[0,0]:
            epoch -= period
        ttime = epoch+period
        while ttime < lc[-1,0]:
            p.plot([ttime,ttime],[np.min(lc[:,1])-0.001,np.max(lc[:,1])+0.001],'r--',linewidth=2.5)
            ttime += period
            
def BinLC(data, NBins):
    '''Allows a lightcurve (in format times, flux, errs) to be binned. Can be performed into times (TimeBin=1, set as secs) or into bin widths (TimeBin=0). 
    The output timebin is an average of the input times.
    Returns array in form 'Bins, flux, err' '''
    #Removing NaNs
    #Array range from start to end, in intervals of 'min' or N (depending which is smaller). 
    binned = np.zeros((NBins, 3))
    bins = np.digitize(data[:, 0], np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), NBins))
    for x in range(NBins):
        if len(data[bins==x, 0])>=1:
            binned[x, :] = np.array([np.average(data[bins==x, 0]), np.average(data[bins==x, 1]), np.std(data[bins==x, 2])])
    binned = binned[binned[:,0]>0,:]
    return binned

def ReadLC(lcdir, TIC):
    #print fname
    if os.path.exists(os.path.join(lcdir,str(TIC)+'_lc.txt')):
        a = np.genfromtxt(os.path.join(lcdir,str(TIC)+'_lc.txt'))
        time = a[:,0]
        flux = a[:,1]
        flux_err = a[:,2]
        lc = np.column_stack((time, flux, flux_err))
        lc = lc[~np.isnan(lc.sum(axis=1))]
    elif os.path.exists(os.path.join(lcdir,str(TIC)+'_lc.fits')):
        a = fits.open(os.path.join(lcdir,str(TIC)+'_lc.fits'))
        
        time = (a[1].data['TIME'])
        flux = (a[1].data['FLUX'])
        flux_err = (a[1].data['FLUX_ERR'])
        lc = np.column_stack((time, flux, flux_err))
        lc = lc[~np.isnan(lc.sum(axis=1))]
        meddat = np.median(lc[:,1])
        lc[:, 2] /= meddat
        lc[:, 1] /= meddat
    else:
        print('No LC found')
        return None
        
    #lc = CutAnoms(lc)

    ###cut section in middle with anomaly
    #highindex = np.searchsorted(Lc[:,0],57430.)
    #lowindex = np.searchsorted(Lc[:,0],57429.2)
    #lc = np.concatenate((lc[:lowindex,:],lc[highindex:,:]))
    ###
   
    return lc

def CutAnoms(lc, cut=98):
    errthresh = np.percentile(lc[:,2],cut)
    newlc = lc[lc[:, 2]<errthresh, :]
    return newlc

def Eyeball(newfname, TIC, N, BLSdets, LCdir, pgramfile, TICinfo=None):
    print('#########################')
    print('# TICID:   '+str(TIC)+ '  #')
    lc = ReadLC(LCdir, TIC)
    if lc is not None:
       
        p.figure(1);p.clf()

        run0 = BLSdets[BLSdets['MultiRun']==1]
        period = run0['Per_1']
        epoch = run0['Epoch_1']
        bins = 0.04  #1 hour bins

        PlotLC(lc, 0, '#000066',bins,1,'TESS Lightcurve for '+str(TIC), ephem=(period,epoch))

        p.xlabel('Time (days)');p.ylabel('Relative Flux');p.title('TESS Lightcurve.')
        
        #each row of BLSdets is one periodogram run, and should contain up to 4 peaks. other rows are reruns after removing top peak from previous run

        PowArr = np.genfromtxt(pgramfile)
        
        PlotBLS(lc, BLSdets,  PowArr)
        
        if TICinfo is not None:
            if TIC in TICinfo['TIC']:
                TICdat = TICinfo[TICinfo['TIC']==TIC]
                print('# TESSMAG:     '+str(TICdat['Tessmag'])+ '     #')
                print('# TIC Teff:     '+str(TICdat['Teff'])+ '     #')
                print('#######################')
            else:
                print('No TIC info found')
        p.show(block=False)
        comment = input('ENTER A COMMENT [A>B>C=Planets, E=EB, Nothing:Enter]:')

        if comment!='':
            OutputCSV(newfname, TIC, comment)
    
def FoldLC(lc, period, epoch=0,  opposite=0):
    outlc=np.column_stack(( np.mod((lc[:, 0]-epoch),period)/period, lc[:, 1], lc[:, 2]))
    outlc=outlc[outlc[:, 0].argsort(), :]
    return outlc

def PlotBLS(lc, BLSdetns,  BLSArr, bins=50.0, i=0):
    nruns = np.max(BLSdetns['MultiRun'])
    
    #generate full plot with pgram for run 0
    p.figure(4, figsize=[16.0,6.]); p.clf(); p.title('BLS Folder periodogram');p.xlabel('Period(days)');p.ylabel('Detection Signal')
    p.subplot(2, 1, 1);
    threshold = BLSArr[BLSArr[:,1].argsort(),1][int(np.round(0.98*len(BLSArr[:, 0])))]
    #SNRs=(BLSArr[:, 1]-np.median(BLSArr[BLSArr[:, 1]<threshold, 1]))/np.std(BLSArr[BLSArr[:, 1]<threshold, 1])
    plotline=[np.max(BLSArr[:,1])*0.95, np.min(BLSArr[:,1])*1.2]
    #p.plot([0.241, 0.241], plotline,'--r',  linewidth=2)
    p.plot(BLSArr[:, 0], BLSArr[:, 1], '-',color='#66FFFF')
    
    run0 = BLSdetns[BLSdetns['MultiRun']==1]
    peaks = run0[['Per_1','Per_2','Per_3']].view(np.float64)
    peak_fvals = run0[['Fout_1','Fout_2','Fout_3']].view(np.float64)
    peak_epochs = run0[['Epoch_1','Epoch_2','Epoch_3']].view(np.float64)
    
    p.plot(peaks, peak_fvals, 'o',color='#3333FF')
    p.xscale('log')
    p.title('BLS Plot')

    for a in np.arange(3):
        subplot(peaks[a], peak_fvals[a], peak_epochs[a], lc, bins, a)
        
    #generate phasefold plots for 2 max other runs
    for a in np.arange(2)+1:

        p.figure(4+a); p.clf(); 
        if np.sum(BLSdetns['MultiRun']==a+1)>0:
            row = BLSdetns[BLSdetns['MultiRun']==a+1]
            period = row['Per_1']
            epoch = row['Epoch_1']
            p.title('Run '+str(a)+'. P: '+str(np.round(period, 4))+'. E: '+str(np.round(epoch, 4)))
            foldLc=FoldLC(lc, period, epoch=epoch+period/2.) #sets transit at phase 0.5
            foldLc=foldLc[foldLc[:, 1]>0.01, :]
            binlc=BinLC(foldLc, int(bins))
            binlc=binlc[binlc[:, 1]>0.25, :]
            p.xlim([0.0, 1.00])
            col='#'+"".join([random.choice('01234567') for n in range(6)])
            p.errorbar(foldLc[:, 0], foldLc[:, 1], yerr=foldLc[:, 2], fmt='.',color=col)
            col='#'+"".join([random.choice('01234567') for n in range(6)])
            p.errorbar(binlc[:, 0], binlc[:, 1], yerr=binlc[:, 2], fmt='.',color=col)
        
    p.show(block=False)

def subplot(period, fval, epoch, lc, bins, a):
    '''Plotting to one of 3 subplots the best-fit period.'''
    col= '#'+"".join([random.choice('456789abcdef') for n in range(6)])
    p.subplot(2, 1, 1);p.plot([period,period], [0, 1.2*fval], color=col,  linewidth=2.5)
    p.subplot(2, 3, a+4)
    foldLc=FoldLC(lc, period, epoch=epoch+period/2.) #sets transit at phase 0.5
    foldLc=foldLc[foldLc[:, 1]>0.01, :]
    binlc=BinLC(foldLc, int(bins))
    binlc=binlc[binlc[:, 1]>0.25, :]

    p.ion(); p.figure(4); p.title(str(a)+'. P: '+str(np.round(period, 4)));p.xlabel('Phase');
    p.xlim([0.0, 1.00])
    p.errorbar(foldLc[:, 0], foldLc[:, 1], yerr=foldLc[:, 2], fmt='.',color=col)
    col='#'+"".join([random.choice('01234567') for n in range(6)])
    p.errorbar(binlc[:, 0], binlc[:, 1], yerr=binlc[:, 2], fmt='.',color=col)


def OutputCSV(newfname, TIC,  comment):
    '''This module writes a file for each TIC with comments on the planet likelihood. '''
    comment=comment.replace(',', ';')
    if os.path.exists(newfname):
        fd = csv.writer(open(newfname,'a'))
        fd.writerow([TIC]+[comment])
    else:
        with open(newfname, 'wb') as ca:
            co = csv.writer(ca)
            co.writerow(['TICID','COMMENT'])
            co.writerow([TIC]+[comment])


if __name__ == '__main__':
    Run(int(sys.argv[1]), int(sys.argv[2]))
