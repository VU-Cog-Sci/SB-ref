
#####
#    Created on Oct 15 13:13:11 2019
#
#    @author: inesverissimo
#
#    Do pRF fit on median run, make iterative fit and save outputs
####

import os
import json
import sys
import glob
import re

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import scipy as sp
import scipy.stats as stats
import nibabel as nb
from nilearn.image import mean_img

from nilearn import surface

from utils import *  # import script to use relevante functions

# requires pfpy be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.grid import Iso2DGaussianGridder
from prfpy.fit import Iso2DGaussianFitter


# define participant number and open json parameter file
if len(sys.argv) < 2:
    raise NameError('Please add subject number (ex:1) '
                    'as 1st argument in the command line!')

elif len(sys.argv) < 3:
    raise NameError('Please select server being used (ex: aeneas or cartesius) '
                    'as 2nd argument in the command line!')

else:
    # fill subject number with 0 in case user forgets
    sj = str(sys.argv[1]).zfill(2)


json_dir = '/home/inesv/SB-ref/scripts/analysis_params.json' if str(
    sys.argv[2]) == 'cartesius' else 'analysis_params.json'

with open(json_dir, 'r') as json_file:
    analysis_params = json.load(json_file)

# use smoothed data?
with_smooth = analysis_params['with_smooth']


# define paths and list of files
if str(sys.argv[2]) == 'cartesius':
    filepath = glob.glob(os.path.join(
        analysis_params['post_fmriprep_outdir_cartesius'], 'prf', 'sub-{sj}'.format(sj=sj), '*'))
    print('functional files from %s' % os.path.split(filepath[0])[0])
    out_dir = analysis_params['pRF_outdir_cartesius']

elif str(sys.argv[2]) == 'aeneas':
    print(os.path.join(
        analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=sj), '*'))
    filepath = glob.glob(os.path.join(
        analysis_params['post_fmriprep_outdir'], 'prf', 'sub-{sj}'.format(sj=sj), '*'))
    print('functional files from %s' % os.path.split(filepath[0])[0])
    out_dir = analysis_params['pRF_outdir']

# changes depending on data used
if with_smooth == 'True':
    # last part of filename to use
    file_extension = '_sg_psc_smooth%d.func.gii' % analysis_params['smooth_fwhm']
    # compute median run, per hemifield
    median_path = os.path.join(
        out_dir, 'sub-{sj}'.format(sj=sj), 'run-median', 'smooth%d' % analysis_params['smooth_fwhm'],'iterative_fit')
else:
    # last part of filename to use
    file_extension = '_sg_psc.func.gii'
    # compute median run, per hemifield
    median_path = os.path.join(out_dir, 'sub-{sj}'.format(sj=sj), 'run-median','iterative_fit')

# list of functional files
filename = [run for run in filepath if 'prf' in run and 'fsaverage' in run and run.endswith(
    file_extension)]
filename.sort()
if not os.path.exists(median_path):  # check if path to save median run exist
    os.makedirs(median_path)


med_gii = []
for field in ['hemi-L', 'hemi-R']:
    hemi = [h for h in filename if field in h]

    # set name for median run (now numpy array)
    med_file = os.path.join(median_path, re.sub(
        'run-\d{2}_', 'run-median_', os.path.split(hemi[0])[-1]))
    # if file doesn't exist
    if not os.path.exists(med_file):
        med_gii.append(median_gii(hemi, median_path))  # create it
        print('computed %s' % (med_gii))
    else:
        med_gii.append(med_file)
        print('median file %s already exists, skipping' % (med_gii))


# create/load design matrix
png_path = '/home/inesv/SB-ref/scripts/imgs/' if str(
    sys.argv[2]) == 'cartesius' else analysis_params['imgs_dir']
png_filename = [os.path.join(png_path, png) for png in os.listdir(png_path)]
png_filename.sort()

dm_filename = os.path.join(os.getcwd(), 'prf_dm_square.npy')

if not os.path.exists(dm_filename):  # if not exists
    screenshot2DM(png_filename, 0.1,
                  analysis_params['screenRes'], dm_filename,dm_shape = 'square')  # create it
    print('computed %s' % (dm_filename))

else:
    print('loading %s' % dm_filename)

prf_dm = np.load(dm_filename)
prf_dm = prf_dm.T  # swap axis for popeye (x,y,time)


# define model params
fit_model = analysis_params["fit_model"]

TR = analysis_params["TR"]

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm=analysis_params["screen_width"], 
                         screen_distance_cm=analysis_params["screen_distance"], 
                         design_matrix=prf_dm, 
                         TR=TR)

# sets up stimulus and hrf for this gridder
gg = Iso2DGaussianGridder(stimulus=prf_stim,
                          hrf=None,
                          filter_predictions=False,
                          window_length=analysis_params["sg_filt_window_length"],
                          polyorder=analysis_params["sg_filt_polyorder"],
                          highpass=True)

# set grid parameters
grid_nr = analysis_params["grid_steps"]#20
max_ecc_size = analysis_params["max_eccen"]#16
sizes, eccs, polars = max_ecc_size * np.linspace(0.25,1,grid_nr)**2, \
                    max_ecc_size * np.linspace(0.1,1,grid_nr)**2, \
                        np.linspace(0, 2*np.pi, grid_nr)

for gii_file in med_gii:
    print('loading data from %s' % gii_file)
    data = np.array(surface.load_surf_data(gii_file))

    gf = Iso2DGaussianFitter(data=data, gridder=gg, n_jobs=16, fit_css=False)

    #filename for the numpy array with the estimates of the grid fit
    grid_estimates_filename = gii_file.replace('.func.gii', '_estimates.npz')

    if not os.path.isfile(grid_estimates_filename): # if estimates file doesn't exist
        print('%s not found, fitting grid'%grid_estimates_filename)
        # do grid fit and save estimates
        gf.grid_fit(ecc_grid=eccs,
                    polar_grid=polars,
                    size_grid=sizes)

        np.savez(grid_estimates_filename,
              x = gf.gridsearch_params[..., 0],
              y = gf.gridsearch_params[..., 1],
              size = gf.gridsearch_params[..., 2],
              betas = gf.gridsearch_params[...,3],
              baseline = gf.gridsearch_params[..., 4],
              ns = gf.gridsearch_params[..., 5],
              r2 = gf.gridsearch_params[..., 6])


    loaded_gf_pars = np.load(grid_estimates_filename)

    gf.gridsearch_params = np.array([loaded_gf_pars[par] for par in ['x', 'y', 'size', 'betas', 'baseline','ns','r2']]) 
    gf.gridsearch_params = np.transpose(gf.gridsearch_params)

    # do iterative fit
    print('doing iterative fit')
    gf.iterative_fit(rsq_threshold=0.1, verbose=False)

    iterative_out = gii_file.replace('.func.gii', '_iterative_output.npz')
    np.savez(iterative_out,
             it_output=gf.iterative_search_params)

    # do iterative fit again, now with css, n=1 (isn't that just gaussian?)
    print('doing iterative fit with css ')
    gf.fit_css = True
    gf.iterative_fit(rsq_threshold=0.1, verbose=False)

    iterative_css_out = gii_file.replace('.func.gii', '_iterative_css_output.npz')
    np.savez(iterative_css_out,
             it_output=gf.iterative_search_params)

