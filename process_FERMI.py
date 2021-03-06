"""
Created on Tue, Jun 30 2020

FERMI specific helper functions

@author:
    Michael Schneider, MBI Berlin
    Kathinka Gerlinger, MBI Berlin
"""


import numpy as np
import os
import glob
import h5py
import pandas as pd


# commonly used hdf5 entries
mnemonics = dict(
    xgm_UH='photon_diagnostics/FEL01/I0_monitor/iom_uh_a',
    xgm_SH='photon_diagnostics/FEL01/I0_monitor/iom_sh_a',
    filter_UH='photon_diagnostics/UH_Filter/Instrument',
    filter_seed='photon_diagnostics/Seed_Filter/Instrument',
    # filter_EH='photon_diagnostics/Filters/EH_filter_IN',
    # filter_UH='photon_diagnostics/Filters/UH_filter_IN',
    wavelength='photon_source/FEL01/wavelength',
    diode_I0='DATA/FQPDIa',
    diode_sum='DATA/FQPDSum',
    polarization='photon_source/FEL01/polarization_status',
    harmonic='photon_source/FEL01/harmonic_number',
    gatt_pressure='photon_diagnostics/Gas_Attenuator/Pressure',
    gatt='DPI/Attenuator',
    image='image/ccd1',
    delay='SLU/DelayLine',
    alignz='DPI/AlignZ',
    ccdz='DPI/CCDZ',
    samplez='DPI/SampleZ',
    valve_pos2='FEL/ValvePOS2',
    valve_dpi3='FEL/ValveDPI3',
    comment='ExperimentalComments',
)

### glob files
def findfile(basefolder, sample, membrane, bunchid):
    pattern = '%s/rawdata/*%s.h5' % (membrane, bunchid)
    pattern = os.path.join(basefolder, pattern)    
    fname = glob.glob(pattern)
    if len(fname) == 0:
        print('no file found for BID %s on membrane %s!' % (bunchid, membrane))
        print(pattern)
        return
    if len(fname) > 1:
        print('found %d files for bunchid %s on membrane %s.'
              % (len(fname), bunchid, membrane))
        return fname
    return fname[-1]


def get_exp_dataframe(folder, recursive=True, keys={}):
    '''Loads basic experimental data from a list of hdf5 files.
    
    Returns a pandas Dataframe.
    '''
    pattern = '**/*h5' if recursive else '*h5'
    flist = glob.glob(os.path.join(folder, pattern), recursive=recursive)
    if len(flist) == 0:
        raise FileNotFoundError('No files found using pattern ', pattern)
    exp = []
    exp_keys = mnemonics.copy()
    exp_keys.update(keys)
    skip_keys = ['image']  # don't load these
    for h5file in flist:
        try:
            with h5py.File(h5file, mode='r') as h5:
                info = dict(filename=h5file)
                for k, v in exp_keys.items():
                    if k not in skip_keys:
                        try:
                            val = h5[v][()]
                            val = val.decode() if isinstance(val, bytes) else val
                            info.update({k: val})
                        except KeyError:
                            pass
                            # info.update({k: np.nan})
                exp.append(info)
        except:
            raise
    exp = pd.DataFrame(exp)
    exp.filename = exp.filename.apply(str)
    return exp


def loadh5(filename, extra_keys=[], ccd=True, on_error='pass'):
    '''
    Loads CCD image and bunch energy E_tot(µJ).
    
    Parameters
    ==========
    filename : str
        hdf file path/name
       
    extra_keys : list, optional
        list of mnmemonics or hdf data paths
       
    ccd : boolean, optional (default True)
        whether to return the CCD image
    
    on_error : str, optional (default 'pass')
        ignore KeyErrors in hdf file ('raise', default) or throw error ('raise')
        
    Returns
    =======
    image : np.array
        the CCD image (only if ccd=True)
    
    meta : dict
        non-image data
    
    Notes
    =====
    FERMI's GMD is upstream of the last filter (seed filter). The shot-energy
    needs to be corrected for the filter transmission at the experiment's
    wavelength.
    '''
    _h5_CCDpath = mnemonics['image']
    _h5_I0path = 'photon_diagnostics/FEL01/I0_monitor/iom_sh_a'
    meta = {}
    with h5py.File(filename, 'r') as h5file:
#        meta.update({'I0M': np.array(h5file[_h5_I0path], dtype=np.float)})
#        seed = h5file[_h5_seed_filter][()]
        if ccd:
            image = h5file[_h5_CCDpath][()]
        I0M = h5file[_h5_I0path][()]
        meta.update({'I0M': I0M})
        for k in extra_keys:
            try:
                h5path = mnemonics[k] if k in mnemonics else k
                meta.update({k: h5file[h5path][()]})
            except KeyError:
                meta.update({key: np.nan})
                if on_error == 'raise':
                    raise
                elif on_error == 'pass':
                    pass
                else:
                    print('Error reading key ', key, ' - skipping.')
    if ccd:
        return image, meta
    else:
        return meta




class AzimuthalIntegrator(object):
    def __init__(self, imageshape, dist, center, polar_range, pxsize=13.5e-6, rmin=0, nint=100):
        '''
        Create a reusable integrator for repeated azimuthal integration of similar
        images. Calculates array indices for a given parameter set that allows
        fast recalculation.
        
        Parameters
        ==========
        imageshape : tuple of ints
            The shape of the images to be integrated over.
        
        dist : float
            distance from sample to detector plane center in meter
        
        center : tuple of ints
            center coordinates in pixels
        
        polar_range : tuple of ints
            start and stop polar angle (in degrees) to restrict integration to wedges
        
        
        # tilt : float, optional (default 0)
        #     Horizontal tilt angle of the CCD.
        
        pxsize : float, optional (default 13.5e-6)
            Size of a single detector pixel in meter
        
        rmin : int, optional (default 0)

        nint: int, optional (default 100)
            number of intervals
            
        
        Returns
        =======
        ai : azimuthal_integrator instance
            Instance can directly be called with image data:
            > az_intensity = ai(image)
            radial distances and the polar mask are accessible as attributes:
            > ai.distance
            > ai.polar_mask
        '''
        self.shape = imageshape
        cx, cy = center
        sx, sy = imageshape
        xcoord, ycoord = np.ogrid[:sx, :sy]
        xcoord = xcoord - cx
        ycoord = ycoord - cy

        # distance from center
        dist_array = np.hypot(xcoord, ycoord)

        # array of polar angles
        tmin, tmax = np.deg2rad(np.sort(polar_range)) % np.pi
        polar_array = np.arctan2(xcoord, ycoord)
        polar_array = np.mod(polar_array, np.pi)
        self.polar_mask = (polar_array > tmin) * (polar_array < tmax) * (dist_array > rmin)

        maxdist = np.round(np.max([sx-cx, cx, sy-cy, cy])).astype(int)

        # linear q bins
        theta_max = np.arctan(pxsize * maxdist / dist)
        dth = np.linspace(0, theta_max, nint)
        r_bin = np.round(np.tan(dth) * dist / pxsize).astype(int)

        ix, iy = np.indices(dimensions=(sx, sy))
        self.index_array = np.ravel_multi_index((ix, iy), (sx, sy))

        self.theta = np.array([])
        self.flat_indices = []
        for d in range(nint - 1):
            ring_mask = self.polar_mask * (dist_array >= r_bin[d]) * (dist_array < r_bin[d + 1])
            self.flat_indices.append(self.index_array[ring_mask])
            self.theta = np.append(self.theta, dth[d])
    
    def __call__(self, image):
        assert self.shape == image.shape, 'image shape does not match'
        image_flat = image.flatten()
        return np.array([np.nansum(image_flat[indices]) for indices in self.flat_indices])
