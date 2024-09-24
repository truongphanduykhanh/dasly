# -*- coding: utf-8 -*-
"""
simpleDASreader v4.0 - OptoDAS raw file v7 reader for Python. 
This code requires the library h5pydict.py available from ASN. 
User documentation of the raw data format is available in the 
ASN document OPT-TN1287-R03.
"""

import numpy as np 
import datetime
import os 
from sympy import symbols,sympify

from dasly.simpledas import h5pydict

      
def combine_units(units, operator='*'):
    """
    Combines units from a list of strings by chosen operator.
   
    Parameters
    ----------
    units : list
        List of strings. Units to combine by operator.
    operator: str
        String deciding what operator to combine units by.
        '*' => multiplication (default)
        '/' => division
    Returns
    -------------
    combinedUnit: str
        The combined unit.
    """
    knownUnits = ('rad','dt','strain','W','NormCts','m','K','s','ε')
    usyms = symbols(knownUnits)
    udict = {u:s for u,s in zip(knownUnits,usyms)}
    
    combinedUnit=sympify(units[0], udict)
    for unit in units[1:]:
        if operator=='/': combinedUnit/=sympify(unit,udict)
        if operator=='*': combinedUnit*=sympify(unit,udict)
        
    return str(combinedUnit)


def unwrap(phi, wrapStep=2*np.pi, axis=-1):
    """
    Unwrap phase phi by changing absolute jumps greater than wrapStep/2 to
    their wrapStep complement along the given axis. By default (if wrapStep is
    None) standard unwrapping is performed with wrapStep=2*np.pi.
    
    (Note: np.unwrap in the numpy package has an optional discont parameter
    which does not give an expected (or usefull) behavior when it deviates
    from default. Use this unwap implementation instead of the numpy
    implementation if your signal is wrapped with discontinuities that deviate
    from 2*pi.)
    """
    scale = 2*np.pi/wrapStep
    return (np.unwrap(phi*scale, axis=axis)/scale).astype(phi.dtype)

def wrap(x, wrapStep=2*np.pi):
    """ 
    Inverse of the unwrap() function.
    Wraps x to the range [-wrapStep/2, wrapStep/2>. 
    """
    if wrapStep>0:
        return (x + wrapStep/2)%wrapStep - wrapStep/2
    else:
        return x

def format_time(seconds,fmt="%Y%m%d %H:%M:%S"):
    """
    Convert seconds since Unix epoch time 1970-01-01 T00:00:00Z (UTC) to 
    readable string.
    """
    dt = datetime.datetime.utcfromtimestamp(float(seconds))
    return dt.strftime(fmt)

def _absolute_channels(meta):
    """
    Returns a 1D-array of absolute channel numbers given the ROI provided by 
    meta. Not needed for filversion 7 or greater since
    meta['header']['channels'] can be used directly.
    
    Example:
    --------
    Find the absolute channel of relative channel X:
        
    Example of use:    
        signal, meta = load_DAS_data(FILEPATH)
        absChs=_absolute_channels(meta)
        absoluteChannel= absCh[relativeChannel]
    """
    rois=np.vstack((meta['demodSpec']['roiStart'],
                    meta['demodSpec']['roiEnd'],
                    meta['demodSpec']['roiDec'])).T
    roiAbsChs=[]
    for roi in rois:
        # Each roi has absolute inclusive channel indexing
        roiAbsChs+=[np.arange(roi[0],roi[1]+1,roi[2])] 
    return np.sort(np.unique(np.concatenate(roiAbsChs)))

def _ROI_channels(meta,roiIndex):
    """
    Returns a 1D-array of channel indices in data for selected ROI(s).

    Parameters
    ----------
    meta : dict
        Meta data from DAS file.
    roiIndex: int, list, range or None
        Index of the ROI(s) to get channel indices from. There can be a maximum
        of 8 ROIs in a DAS file. Thus, roiIndex can never be more than 7.
    """
    channels = meta['header']['channels']
    chsOut = []
    rois=np.vstack((meta['demodSpec']['roiStart'],
                    meta['demodSpec']['roiEnd'],
                    meta['demodSpec']['roiDec'])).T

    if isinstance(roiIndex,int):
        roiIndex = [roiIndex]
    
    for n in roiIndex:
        if n < rois.shape[0]:
            achs = np.intersect1d(channels,
                                  np.arange(rois[n,0],rois[n,1]+1,rois[n,2]))
            chs = np.arange(len(channels))[np.in1d(channels,achs)]
            chsOut.append(chs)
    
    return np.hstack(chs)

def _fix_meta(meta):
    """
    For backward compatibility.
    Updates old meta dict to version 7.
    
    Parameters
    ----------
    meta : dict
        meta dict returned from load_DAS_file().
    """
    if meta['fileVersion']<7:
        c=299792458 #speed of light in vacuum
        if not 'cableSpec' in meta.keys():
            meta['cableSpec']={'fiberOverlength':1.0,
                               'refractiveIndex':1.4677,
                               'zeta':0.78}
        dx_fiber = meta['demodSpec']['dTau']*c\
            /(2*meta['cableSpec']['refractiveIndex'])
        
        meta['header']['dx']= dx_fiber/meta['cableSpec']['fiberOverLength']
        
        if not 'gaugeLength' in meta['header'].keys():
            meta['header']['gaugeLength']= meta['demodSpec']['nDiffTau']*dx_fiber
        
        if not 'dataScale' in meta['header'].keys():
            meta['header']['dataScale']=np.pi/2**29
        meta['header']['dataScale']/=meta['header']['dt']\
            *meta['header']['gaugeLength']
        
        if not 'spatialUnwrRange' in meta['header'].keys():
            meta['header']['spatialUnwrRange']=8*np.pi
        meta['header']['spatialUnwrRange']/=meta['header']['dt']\
            *meta['header']['gaugeLength']
        
        meta['header']['unit']='rad/m/s'
        meta['header']['sensitivityUnit']='rad/m/ε'
        channels=_absolute_channels(meta)
        meta['header']['channels']=channels
        
            
            
        meta['cableSpec']['sensorDistances']=channels*meta['header']['dx']
        
        itu=int(meta['monitoring']['Laser']['itu'])
        wavelength=c/(190e12+itu*1e11)
        
        meta['header']['sensitivity']=4*np.pi*meta['cableSpec']['zeta']\
            *meta['cableSpec']['refractiveIndex']/wavelength
        
        meta['header']['sensitivityUnit']='rad/m/ε'
        


def load_DAS_file(filename, chIndex=None, roiIndex=None, samples=None,
                  integrate=True, unwr=True, metaDetail=1, useSensitivity=True,
                  spikeThr=None):
    """
    Load demodulated signal and metadata. 
    
    Parameters
    ----------
    filename: string
        Full path + filename of file to load.
    chIndex: list, range or None
        Channel indices to load.
        Ingored if roiIndex is not None.
        None => load all available channels (default). 
    roiIndex: list, range or None
        Returns all channels of a region of interest (ROI) with indices in
        roiIndex.
        If None, chIndex is used to select channels (default).
    samples: range, int, or None
        Time indices to read. 
        None  => load all available time samples (default). 
        int   => Load this number of samples, starting from first sample.
        range => Range of indices, i.e. samples=range(start,stop) 
        Note: Decimation should be performed after cumsum or antialiasing.
              Decimation with the range.step parameter is not recommended.
    integrate: bool
        Integrate along time axis of phase data (default).
        If false, the output will be the time differential of the phase.       
    unwr: boolean
        Unwrap along spatial axis before cumsum. 
        Defaults to True. Use False for hydrophones or Bragg grating arrays.  
    metaDetail: int
        1 => Load only metadata needed for DAS data interpretation
        2 => Load all metadata
    useSensitivity: bool
        Scale (divide) signal with meta['header']['sensitivity']. 
    spikeThr: float or None
        Threshold (in readians) for spike detection and removal. 
        Samples that exceed this threshold in absolute value before cumsum 
        (or at output if cumsum is disabled) will be set to zero.  
        As default spikeThr=None, and there is no spike detection. 
        
    Returns
    -------
    signal: array(Nt,Nch) of float
        Recorded signal data with time index in first dimension and channel 
        index in the second. 
    meta: dict
        Dictionary with selected metadata. 
        Metadata fields relevant for end users are described in
        'OPT-TN1287 OptoDAS HDF5 file format description for externals.pdf'
        The meta['appended'] includes fields appended by this function:
            dataOffs: array(Nch) of float
                The header['phiOffs'] of the extracted channels, scaled 
                with same factor as data.
            unit: str
                The unit of the output data. If useSensitivity=True, 
                unit = strain else unit = rad.
            absChs: array(Nch) of int
                The absolute channel numbers of the extracted data channels.      
    """
    with h5pydict.DictFile(filename,'r') as f:
        # Load metedata (all file contents except data field)
        m = f.load_dict(skipFields=['data']) 
        if metaDetail==1:
            ds=m['demodSpec']
            mon=m['monitoring']
            meta=dict(fileVersion = m['fileVersion'],
                      header     = m['header'],
                      timing     = m['timing'],
                      cableSpec  = m['cableSpec'],
                      monitoring = dict(Gps = mon['Gps'],
                                        Laser=dict(itu=mon['Laser']['itu'])),
                      demodSpec  = dict(roiStart = ds['roiStart'],
                                        roiEnd   = ds['roiEnd'],
                                        roiDec   = ds['roiDec'],
                                        nDiffTau = ds['nDiffTau'],
                                        nAvgTau  = ds['nAvgTau'],
                                        dTau     = ds['dTau']))
        else:
            meta = m
            
        _fix_meta(meta)
                
        # Express samples as a range
        if isinstance(samples, int):
            samples=range(0, samples)
        elif samples is None:
            samples = slice(None)            
        if isinstance(samples,range) and samples.step>1 and integrate:
            print('Warning: Time decimation before cumsum (samples.step=%d>1)\
                  is not recommended.' %samples.step)
        
        if roiIndex is not None:
            assert chIndex is None, "chIndex must be None when using roiIndex."
            chIndex = _ROI_channels(meta, roiIndex)
        elif chIndex is None:
            chIndex = slice(None)
        
        signal = f['data'][:,chIndex][samples,:]\
            * np.float32(meta['header']['dataScale'])  
                
    if unwr or spikeThr or integrate:
        if meta['header']['dataType']<3 or meta['demodSpec']['nDiffTau']==0:
            raise ValueError('Options unwr, spikeThr or integrate can only be\
                             used with time differentiated phase data')
    if unwr and meta['header']['spatialUnwrRange']:
        signal=unwrap(signal,meta['header']['spatialUnwrRange'],axis=1) 
    
    if spikeThr is not None:
        signal[np.abs(signal)>spikeThr] = 0
    
    unit=meta['header']['unit'] 
    if integrate:
        signal=np.cumsum(signal,axis=0)*meta['header']['dt']
        unit=combine_units([unit, unit.split('/')[-1]]) 
       
    if useSensitivity:
        signal/=meta['header']['sensitivities']
        unit=combine_units([unit, meta['header']['sensitivityUnits']],'/')

    meta.update(appended = dict(
                dataOffs=meta['header']['phiOffs'][chIndex],
                unit = unit,
                channels = meta['header']['channels'][chIndex]
                ))
    
    return signal, meta

def load_multiple_DAS_files(path, fileIds, chIndex=None, roiIndex=None,
                            integrate=True, unwr=True, metaDetail=1,
                            useSensitivity=True, spikeThr=None):       
    """cumsum
    Load and concatenate multiple DAS files. Files shoud be from a contigous
    recording.
    
    Inputs
    ======
    path: string
        Path to folder containing DAS files to load.
    fileIds: list of int
        Integer filenames (without '.hdf5' extension) of data to load.
    chIndex: list, range or None
        Channel indices to load.
        Ingored if roiIndex is not None.
        None => load all available channels (default). 
    roiIndex: list, range or None
        Returns all channels of a region of interest (ROI) with indices in
        roiIndex.
        If None, chIndex is used to select channels (default).
    integrate: bool
        Integrate along time axis of phase data (default).
        If false, the output will be the time differential of the phase.       
    unwr: boolean
        Unwrap along spatial axis before cumsum. 
        Defaults to True. Use False for hydrophones or Bragg grating arrays.  
    metaDetail: int
        1 => Load only metadata needed for DAS data interpretation
        2 => Load all metadata
    useSensitivity: boolcumsum
        Scale (divide) signal with meta['header']['sensitivity']. 
    spikeThr: float or None
        Threshold (in readians) for spike detection and removal. 
        Samples that exceed this threshold in absolute value before cumsum 
        (or at output if cumsum is disabled) will be set to zero.  
        As default spikeThr=None, and there is no spike detection.
        
    Returns
    -------
    concatSig: array(Nt,Nch) of float
        Recorded signal data with time index in first dimension and channel 
        index in the second. 
    meta: dict
        Dictionary with selected metadata. 
        Note: meta['header']['phiOffs'] will be reduced to the requested
        channels. 
        Metadata fields relevant for end users are described in
        'OPT-TN1287 OptoDAS HDF5 file format description forcumsum externals.pdf'
        The meta['appended'] includes fields appended by this function:
            dataOffs: array(Nch) of float
                The header['phiOffs'] of the extracted channels, scaled 
                with same factor as data.
            unit: str
                The unit of the output data. If useSensitivity=True, 
                unit = strain else unit = rad.
            channels: array(Nch) of int
                The absolute channel numbers of the extracted data channels.
    """
    print('Loading files: ', end='')
    for fileId in fileIds:
        filename=os.path.join(path, str(fileId).zfill(6)+'.hdf5')
        signal, m = load_DAS_file(filename,
                                  chIndex=chIndex,
                                  roiIndex = roiIndex,
                                  samples=None,
                                  integrate=False,
                                  unwr=False,
                                  spikeThr=None,
                                  metaDetail=1,
                                  useSensitivity=useSensitivity)
        if fileId == fileIds[0]:
            meta=m
            unit=meta['appended']['unit']
            concatSig=signal
        else:
            concatSig=np.concatenate((concatSig,signal),axis=0)
        print('%d, '%fileId, end='')
    print('Completed.')

    if unwr or spikeThr or integrate:
        if meta['header']['dataType']<3 or meta['demodSpec']['nDiffTau']==0:
            raise ValueError('Options unwr, spikeThr or integrate can only be\
                             used with time differentiated phase data')
    if unwr and meta['header']['spatialUnwrRange']:
        concatSig=unwrap(concatSig,meta['header']['spatialUnwrRange'],axis=1)
    if spikeThr is not None:
        concatSig[np.abs(concatSig)>spikeThr] = 0
    if integrate:
        concatSig=np.cumsum(concatSig,axis=0)*meta['header']['dt']
        unit=combine_units([unit, unit.split('/')[-1]]) 
    meta['appended']['unit']=unit

    return concatSig, meta

if __name__=='__main__':
    """
    A description of the OptoDAS HDF5 file format is available in:
    'OPT-TN1287-R03 OptoDAS HDF5 file format description for externals.pdf'
    """
    import pylab as plt
    
    filename='<path + filename of raw DAS data file>'
    filename='/raid1/fsi/exps/ver7_testing/20210303/dphi/130344.hdf5'
    
    # load data in unit ε
    # to load data in rad/m set useSensitivty=False
    # to load data in rad set useSensitivty=False, and mutiply with gaugelength:
    # signal *= meta['header']['gaugeLength']
    # meta['appended']['unit'] = combine_units({meta['appended']['unit'],'m','*')
    signal, meta = load_DAS_file(filename,
                                 chIndex=range(20,150,10),
                                 #roiIndex=0,
                                 integrate=True,
                                 samples=None,
                                 useSensitivity=True,
                                 unwr=True
                                 )
    
    
    # Get positions in meters
    positions=meta['appended']['channels']*meta['header']['dx']
            
    # Get time axis
    time=np.arange(signal.shape[0])*meta['header']['dt']
    
    # Plot time-series of selected channels
    plt.figure('time-series')
    
    absChs=[100, 240]
    
    for absCh in absChs:
        index=np.argmax(positions>=absCh)
        plt.plot(time,
                 signal[:,index],
                 label='%.1f m'%positions[index])
        plt.title(format_time(meta['header']['time']))
        plt.ylabel(meta['appended']['unit'])
        plt.xlabel('Time (s)')
        plt.legend()
        plt.show()
    