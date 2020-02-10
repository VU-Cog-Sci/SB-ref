# change way of getting median estimates
# (now I'm giving all subs same weight for all voxels)
# so trying to get a more precise account of RF

import os, json
import sys, glob
import numpy as np

from utils import *

import matplotlib.pyplot as plt
from nilearn import surface
from distutils.util import strtobool

import cortex
import seaborn as sns
import pandas as pd


# requires pfpy be installed - preferably with python setup.py develop
from prfpy.rf import *
from prfpy.timecourse import *
from prfpy.stimulus import PRFStimulus2D
from prfpy.grid import Iso2DGaussianGridder,CSS_Iso2DGaussianGridder
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

from popeye import utilities 

import matplotlib.gridspec as gridspec
import scipy

import matplotlib.patches as patches
from statsmodels.stats import weightstats

import random

from joblib import Parallel, delayed


sj = 'median'

with open('analysis_params.json','r') as json_file: 
            analysis_params = json.load(json_file)  


# used smoothed data (or not) for plots
with_smooth = 'False'#analysis_params['with_smooth']

# fit model to use (gauss or css)
fit_model = 'css' #analysis_params["fit_model"]
# if using estimates from iterative fit
iterative_fit = True #True

# total number of chunks that were fitted (per hemi)
total_chunks = analysis_params['total_chunks']


all_estimates = append_pRFestimates(analysis_params['pRF_outdir'],
                          model=fit_model,iterative=iterative_fit,exclude_subs=['sub-07'],total_chunks=total_chunks)


# load all
rsq = np.array(all_estimates['r2'])
ns = np.array(all_estimates['ns'])
betas = np.array(all_estimates['betas'])
baseline = np.array(all_estimates['baseline'])
xx = np.array(all_estimates['x'])
yy = np.array(all_estimates['y'])
size = np.array(all_estimates['size'])


# set limits for xx and yy, forcing it to be within the screen boundaries

vert_lim_dva = (analysis_params['screenRes'][-1]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][-1])
hor_lim_dva = (analysis_params['screenRes'][0]/2) * dva_per_pix(analysis_params['screen_width'],analysis_params['screen_distance'],analysis_params['screenRes'][-1])

# make new variables that are masked (center of RF within screen limits and only positive pRFs)
# also max size of RF is 10 dva
print('masking variables to be within screen and only show positive RF')

masked_xx = []
masked_yy = []
masked_size = []
masked_beta = []
masked_baseline = []
masked_rsq = []
masked_ns = []

for i in range(len(all_estimates['subs'])):
    new_estimates = mask_estimates(xx[i],yy[i],size[i],betas[i],baseline[i],rsq[i],vert_lim_dva,hor_lim_dva,ns=ns[i])
    
    masked_xx.append(new_estimates['x']) 
    masked_yy.append(new_estimates['y'])
    masked_size.append(new_estimates['size'])
    masked_beta.append(new_estimates['beta'])
    masked_baseline.append(new_estimates['baseline'])
    masked_rsq.append(new_estimates['rsq'])
    masked_ns.append(new_estimates['ns'])


##### load prfpy classes to get single prediction #############

# create/load design matrix
png_path = analysis_params['imgs_dir']
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
prf_dm = prf_dm.T # then it'll be (x, y, t)

# change DM to see if fit is better like that
# do new one which is average of every 2 TRs

prf_dm = shift_DM(prf_dm)
prf_dm = prf_dm[:,:,analysis_params['crop_pRF_TR']:] # crop DM because functional data also cropped now

# define model params
TR = analysis_params["TR"]
hrf = utilities.spm_hrf(0,TR)

# make stimulus object, which takes an input design matrix and sets up its real-world dimensions
prf_stim = PRFStimulus2D(screen_size_cm=analysis_params["screen_width"], 
                         screen_distance_cm=analysis_params["screen_distance"], 
                         design_matrix=prf_dm, 
                         TR=TR)



# divide and save chunks in folder, avoids memory issues
out_dir = os.path.join(analysis_params['derivatives'],'tests_average_RF')
if not os.path.exists(out_dir):  # check if path exists
    os.makedirs(out_dir)

chunk_num = find_zero_remainder(np.array(masked_xx).shape[-1],100)[-1] #94 chunks
num_vox_chunk = int(np.array(masked_xx).shape[-1]/chunk_num) # total number of vertices per chunks

for w in range(chunk_num):

    filename = 'average_RF_chunk-%s.npz'%str(w).zfill(2)

    if os.path.isfile(os.path.join(out_dir,filename)): # if file already exists

      RF = np.load(os.path.join(out_dir,filename)) # load
      RF = RF['avg_RF']

      # smooth the RF
      avg_RF = Parallel(n_jobs=16)(delayed(smooth_RF)(RF[i,:,:],sigma=20) for i in range(RF.shape[0]))
      filename = 'average_RF_smooth_chunk-%s.npz'%str(w).zfill(2)

    else:

      avg_RF = Parallel(n_jobs=16)(delayed(combined_rf)(prf_stim,np.array(masked_xx)[:,i],np.array(masked_yy)[:,i],
                           np.array(masked_size)[:,i],np.array(masked_rsq)[:,i],np.array(masked_ns)[:,i]) 
                         for i in range(num_vox_chunk*w,num_vox_chunk*(w+1)))#np.array(masked_xx).shape[-1]))

    np.savez(os.path.join(out_dir,filename), avg_RF = avg_RF)

# compute polar angle value  for those RF
    
for w in range(chunk_num):

    RF_filename = 'average_RF_chunk-%s.npz'%str(w).zfill(2)
    loaded_RF = np.load(os.path.join(out_dir,RF_filename))
    loaded_RF = loaded_RF['avg_RF']

    avg_pa = Parallel(n_jobs=16)(delayed(max_RF2PA)(loaded_RF[i,:,:]) for i in range(loaded_RF.shape[0]))
    np.savez(os.path.join(out_dir,'average_PA_chunk-%s.npz'%str(w).zfill(2)), avg_pa = avg_pa)

    # repetitive but ok for now
    RF_filename = 'average_RF_smooth_chunk-%s.npz'%str(w).zfill(2)
    loaded_RF = np.load(os.path.join(out_dir,RF_filename))
    loaded_RF = loaded_RF['avg_RF']

    avg_pa = Parallel(n_jobs=16)(delayed(max_RF2PA)(loaded_RF[i,:,:]) for i in range(loaded_RF.shape[0]))
    np.savez(os.path.join(out_dir,'average_PA_smooth_chunk-%s.npz'%str(w).zfill(2)), avg_pa = avg_pa)



























