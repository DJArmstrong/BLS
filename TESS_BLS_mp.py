# -*- coding: utf-8 -*-
'''

'''

import numpy as np
import csv
import sys
import os
import glob
from astropy.io import fits
import bls
import TESSselfflatten as tsf
import logging
import subprocess

logging.basicConfig(filename='BLSrun.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

'''CSV file in = KIC id. RA. Dec. Vmag. Median Flux, Tcen,Tdur, depth,SNR,SNR(RN),GoodnessOfFit'''

def mpmain(indirectory, exofopdat=None, filename='TESS_detns_full.csv', mp=False):
    flist=glob.glob(os.path.join(indirectory,'tess*.fits'))
    if mp:
        #multiprocessing
        from pathos.multiprocessing import ProcessingPool as Pool
        import multiprocessing
        ncores=multiprocessing.cpu_count()
        logging.debug('Running multiprocessing with '+str(ncores)+' cores')

        #Splitting the old fashioned way by cutting a list up:
        
        splits=np.ceil(np.arange(0,len(flist),len(flist)/float(ncores))).astype(int)
        flists=np.array_split(flist,splits)[1:]
        flists=[list(flists[i]) for i in range(len(flists))]  #now have list of lists

        #Multiprocessing (I hope...)
        pool = Pool(ncores)
        outputs=pool.map(Run,flists,list(np.tile(filename,ncores)),list(np.tile(exofopdat,ncores)),range(ncores))
        ext = filename[-4:]
        outflist = glob.glob(filename[:-4]+'*'+ext)
        outfile = filename[:-4]+'_all.txt' #don't use the same extension as the per-process outfiles
        subprocess.call('(head -1 '+outflist[0]+' ; tail -n +2 -q '+filename[:-4]+'*'+ext+' ) > '+outfile, shell=True)
    else:
        Run(flist,filename,exofopdat)

def Run(flist, filename, exofopdat=None, N=-1):

    #check already used ids
    if os.path.exists(filename) and os.stat(filename).st_size>0:
        usedids = np.genfromtxt(filename,delimiter=',',skip_header=1)[:,0].astype('int')
    else:
        usedids = []
    
    if N!=-1:   #i.e. multiprocessing
        ext = filename[-4:]
        filename=filename.replace(ext,'_'+str(N)+ext)
        
    if exofopdat is not None:
        exofopinfo = np.genfromtxt(exofopdat, delimiter=',', names=True)
    else:
        exofopinfo = None

    logging.debug('Running BLS')       
    
    Kcount = len(usedids)
    for infile in flist:
        lcdat = fits.open(infile)
        if 'TICID' in lcdat[0].header.keys():
            TIC = lcdat[0].header['TICID']
        elif 'TIC' in lcdat[0].header.keys():
            TIC = int(lcdat[0].header['TIC'])
        else:
            TIC = 0
        
        if TIC not in usedids:
            logging.debug('Searching '+str(TIC)+' -   Nlc='+str(Kcount)+'  -   N='+str(N))
            found = False
            if exofopinfo is not None:
                if TIC in exofopinfo['TIC_ID']:
                    found = True
                    BLSSearchTESS(lcdat, TIC, flist, N, filename, exofopinfo[exofopinfo['TIC_ID']==TIC])
            if not found:
                BLSSearchTESS(lcdat, TIC, flist, N, filename)
        Kcount+=1

def BLSSearchTESS(lcdat, TIC, InDirectory, N, filename, info=None, SNRlimit=3.0, maglim=18.4):
    '''Searches TESS lightcurve for transiting planets using BLS.
    '''
    #Getting LC
    try:
        lcurve = np.array([lcdat[1].data['TIME'],
    					lcdat[1].data['PDCSAP_FLUX'],
    					lcdat[1].data['PDCSAP_FLUX_ERR']]).T
    					
    except (NameError,KeyError):
        lcurve = np.array([lcdat[1].data['TIME'],
    					lcdat[1].data['FLUX'],
    					lcdat[1].data['FLUX']]).T
    
        #lcurve = np.array([lcdat[1].data['TIME'],
    #					lcdat[1].data['SAP_FLUX'],
    #					lcdat[1].data['SAP_FLUX']*0.0001]).T
    try:
        qual = lcdat[1].data['QUALITY']
    except KeyError:
        qual = np.zeros(len(lcdat[1].data['TIME']))
        
    lcdat.close()
    lcurve = lcurve[qual==0,:]
    lcurve = lcurve[lcurve[:,1]>0]
    cut = np.isnan(lcurve[:,1]) | np.isnan(lcurve[:,0]) | np.isnan(lcurve[:,2])
    lcurve = lcurve[~cut,:]
    norm = np.median(lcurve[:,1])
    lcurve[:,1] /= norm
    lcurve[:,2] /= norm
    
    t0 = lcurve[0,0]
    lcurve[:,0] -= t0
    
    if info is not None:
        brightness = info['Tess_Mag'][0]
    else:
        brightness = maglim - 0.1
        
    try:
        if np.isnan(brightness):
            brightness=15.555555
    except ValueError:   #catches multiple entries in exofop file
        brightness = brightness.values[0]
        
    if len(lcurve[:,0])>=50 and brightness<maglim:
        logging.debug('Running BLS search of lightcurve')
        try:
            lcurve[:,1] = tsf.TESSflatten(lcurve,kind='poly')
        except:
            logging.debug('Error - Flattening produced error. Continuing with unflattened lcurve')
        
        test = 0  
        try:
        
        #if TIC in [382302241, 235037761, 29857954, 261136679]:
        #if TIC in [52368076]:
            Out = getBLS(lcurve, TIC, filename, t0=t0)   #this is the actual call
            test= 1
        except TypeError:
            test= 0
            logging.debug('Error - No detections returned')
        except ValueError:
            test = 0
            logging.debug('Error - ValueError')

        if test!=0:
            logging.debug("Adding "+str(len(Out))+" detections to file for TIC "+str(TIC)+'. N='+str(N))
            #Adding detection to file...
            #import pylab as p
            #p.ion()
            #p.figure(1)
            #p.clf()
            #p.plot(lcurve[:,0],lcurve[:,1],'b.')
            #print(TIC)
            
            for run in np.arange(int(np.max(Out[:,3])))+1:
                runout = Out[Out[:,3]==run]
            #    print('Iteration: '+str(run))
            #    print(runout[0,:3])
            #    print(runout[1,:3])
            #    print(runout[2,:3])
                AddDetnToFile(runout, brightness, TIC, filename)        
            #p.pause(5)
            #input()
    else:
        logging.debug('Brightness '+str(info['Tess_Mag'])+' not recognised, or too few datapoints')

def CutTransits(lc,transitt0,transitper,phase1, phase2):
    phase = np.mod(lc[:,0]-transitt0,transitper)/transitper    
    intransit = (phase>phase1) & (phase<phase2)
    return lc[~intransit,:]


def getBLS(lc, TIC, filename, t0=0, multirunthresh=5.):
    '''Runs BLS search for transits on lightcurve.
    Returns 3-column array (detns) with the position and height of any peaks, and the 2-column spectrum'''

    #Using custom timerange across full lightcurve/1.1 to 0.4d
    min_freq = 1.1/(lc[-1, 0]-lc[0, 0])
    max_freq = 1./0.4
    freq_spacing = 1e-5
    nfb = int(np.floor((max_freq - min_freq)/freq_spacing))
    nb = 400
    
    count = 0
    while True:

        #remove transit for all but first run
        if count > 0:
            lccut = CutTransits(lccut,epoch,bper,phase1,phase2)
        else:
            lccut = lc.copy()
        
        powOut = bls.eebls(lccut[:, 0],lccut[:, 1],np.zeros(len(lccut[:, 0])),np.zeros(len(lccut[:, 0])),nfb,fmin=min_freq,df=freq_spacing,nb=nb,qmi=0.005,qma=0.15)
        PowArr = np.column_stack((1/(np.arange(min_freq,(min_freq+nfb*freq_spacing),freq_spacing))[0:nfb], powOut[0]))
        PowArr = PowArr[PowArr[:, 0].argsort(), :]

        #Rescaling both BLS and LS such that the median (of values <0.3*the max value ) is at 1 and the position of a peak gives the height above the median
        PowArr[:, 1] = PowArr[:, 1]/np.median(PowArr[:, 1][PowArr[:, 1]<np.percentile(PowArr[:, 1],95)])
        if count==0:
            blsoutfile = filename[:-4]+'_'+str(TIC)+'_'+str(count)+'_pgram.txt' #includes count in case we want to save the others later
            np.savetxt(blsoutfile, PowArr)
            
        detnsOut_lccut = getPeaks(PowArr)
        detnsOut_lccut = detnsOut_lccut[(-detnsOut_lccut[:, 1]).argsort(), :]
        detnsOut_lccut = np.hstack((detnsOut_lccut,np.ones([len(detnsOut_lccut[:,0]),1])+count))
        
        detnmax = np.max(detnsOut_lccut[:,2])
        
        bper = powOut[1]
        bpow = powOut[2]
        depth = powOut[3]
        qtran = powOut[4]
        duration = bper*qtran
        in1 = powOut[5]
        in2 = powOut[6]
        phase1 = in1/float(nb)
        phase2 = in2/float(nb)    
        epoch = lccut[0,0]
        
        epoch_array = np.zeros([len(detnsOut_lccut[:,0]),1])  #hard to easily extract epoch etc for more than just main peak
        epoch_array[0] = (epoch+phase1*bper) + t0
        
        #check if only one transit in middle of lc
        if epoch + (phase1+1)*bper > lccut[-1,0]: 
            detnmax = multirunthresh+1 #force another run, unless we hit max via count
            detnsOut_lccut[0,0] = -10. #set peak period to -10            
            
        if count == 0:
            detnsOut = np.hstack((detnsOut_lccut,epoch_array))
        else:
            detnsOut = np.vstack((detnsOut,np.hstack((detnsOut_lccut,epoch_array))))  
            
        count += 1
            
        if detnmax < multirunthresh:
            break
  
        if count >2:
            break
    
    return detnsOut

    

    #Removing a trend in logspace. Rescaling with median to above zero
    #PowArr[:, 1] = (PowArr[:,1]-0.9*np.polyval(np.polyfit(np.log10(PowArr[:,0]),PowArr[:,1],1),np.log10(PowArr[:,0])))

    #PowArr[:, 1]=PowArr[:,1]-np.polyval(np.polyfit(np.log10(PowArr[:,0]),PowArr[:,1],2),np.log10(PowArr[:,0]))
    #LSout = pgram.fwmls(lc[:, 0], lc[:, 1], lc[:, 2], 2, 0.3)
    #LSarr = np.column_stack((1.0/LSout[0], LSout[1]))


    #LSarr[:, 1] = LSarr[:, 1]/np.median(LSarr[LSarr[:, 1]<(np.max(LSarr[:, 1])*0.3), 1])


    #Finding 6 hour alias from thrusters. Stacking the ratio of this to all other detections to the detections file
    #AliasSNR=np.max(PowArr[abs(PowArr[:, 0]-0.245164)<0.12, 1])
    #detnsOut=np.column_stack((detnsOut, detnsOut[:, 2]/AliasSNR))
    #time,Power,SNR,SNRlombscar,SNRthruster    


def getPeaks(PowArr):
    '''This module takes in a periodogram spectrum and finds the distinct higest peaks (ie those separated by >x%).
    Inputs:
        FreqArr with columnns of period (days) and detection strength (units).
    Returns:
        DetArr with columns of period (days), det strength and SNR (calculated as ratio to median background level). Sorted by height of peaks.'''
    i2=2
    threshold = PowArr[PowArr[:,1].argsort(),1][int(np.round(0.98*len(PowArr[:, 0])))]
    PowArrCut=PowArr[PowArr[:, 1]>threshold, :]
    detns = np.zeros([2,3])
    for i in range(len(PowArrCut[:, 1])):
        #This technique also has the benefit of only taking the last (and highest) point in a long, spread line of values
        #If within 0.01 of previous detected peak...
        if (abs(PowArrCut[i, 0]-(detns[i2-1,0]))<(PowArrCut[i, 0]*0.01)):
            #print 'Found nearby similar detection to '+str(PowArrCut[i, 0])+' at '+str(detnslist[i2-1][0])
            #Either over-writes previous detection (if stronger detection) or does nothing
            if (PowArrCut[i, 1]>detns[i2-1,1]):
                SNR=(PowArrCut[i, 1]-np.median(PowArr[PowArr[:, 1]<threshold, 1]))/np.std(PowArr[PowArr[:, 1]<threshold, 1])
                #SNRls=PowArrCut[i, 1]/(LSOut[np.argmin(LSOut[:, 0]-PowArrCut[i, 0]), 1])
                #SNRls=float(SNR/(LSOut[np.where(LSOut[:, 0]==pl.find_nearest(LSOut[:, 0],PowArrCut[i, 0])[0]), 1]))
                detns[i2-1,:]=np.array(list(PowArrCut[i])+[SNR])#+[SNRls]
        #Else if not close to limits
        else:
            #Adds to file:
            #logging.debug('Found at '+str(PowArrCut[i, 0])+' . Adding to detection array.'
            #Calculating SNR from lowest 95% of periodogram
            SNR=(PowArrCut[i, 1]-np.median(PowArr[PowArr[:, 1]<threshold, 1]))/np.std(PowArr[PowArr[:, 1]<threshold, 1])
            #SNRls=PowArrCut[i, 1]/(LSOut[np.argmin(LSOut[:, 0]-PowArrCut[i, 0]), 1])
            detns = np.vstack((detns,np.array(list(PowArrCut[i])+[SNR])))#+[SNRls]]
            i2+=1

    detns = detns[detns[:, 0]!=0,:]
    return detns

def AddDetnToFile(ds, brightness, TIC, filename):
    '''This module adds the data to file (specified by filename):
    The first contains short-hand data on the possible detection of a transit, eg name, brigthness, time, duration, depth, error, etc
    '''
    while ds.shape[0] < 4: #avoid errors if less than 5 detections
        ds = np.vstack((ds,np.zeros(ds.shape[1])))
    if os.path.exists(filename):
        with open(filename,'a') as fd:
            c = csv.writer(fd)
            c.writerow([TIC,brightness,ds[0,3],ds[0,0],ds[0,4]]+list(ds[0, 1:3])+[ds[1,0],ds[1,4]]+list(ds[1, 1:3])+[ds[2,0],ds[2,4]]+list(ds[2, 1:3])+[ds[3,0],ds[3,4]]+list(ds[3, 1:3]))
    else:
        #Writing header
        with open(filename, "w") as f:
            c = csv.writer(f)
            c.writerow(['TICID','TESSMAG','MultiRun','Per_1','Epoch_1','Fout_1','SNR_1','Per_2','Epoch_2','Fout_2','SNR_2','Per_3','Epoch_3','Fout_3','SNR_3','Per_4','Epoch_4','Fout_4','SNR_4'])
            c.writerow([TIC,brightness,ds[0,3],ds[0,0],ds[0,4]]+list(ds[0, 1:3])+[ds[1,0],ds[1,4]]+list(ds[1, 1:3])+[ds[2,0],ds[2,4]]+list(ds[2, 1:3])+[ds[3,0],ds[3,4]]+list(ds[3, 1:3]))

def str2bool(input):
    if input in ['y','Y','True','true','1']:
        return True
    else:
        return False

if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv)>2:
        mpmain(sys.argv[1], sys.argv[2], sys.argv[3], str2bool(sys.argv[4]))
    else:
        mpmain(sys.argv[1])
